import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    load_belgeler, parcala, chunkla_belgeler, benzerlik,
    indeksle, getir, sor,
)


# ──────────────────────────────────────────────────────
# MOCK embed + llm — deterministik, key gerektirmez.
# Gerçek projede: embed = sentence-transformers/OpenAI, llm = OpenAI (README).
# fake_embed: anahtar-kelime tabanlı sayım vektörü (anlamca yakın metin → yakın vektör).
# ──────────────────────────────────────────────────────
VOCAB = ["pil", "şarj", "kamera", "video", "garanti", "servis",
         "kutu", "kablo", "su", "ip68"]


def fake_embed(metin):
    s = metin.lower()
    return [float(s.count(w)) for w in VOCAB]


def fake_llm(prompt):
    # Gerçekte akıcı cevap üretir; testte bağlamın ilk cümlesini döndürür.
    baglam = prompt.split("Bağlam:")[1].split("Soru:")[0].strip()
    return "Cevap: " + baglam[:40]


@pytest.fixture(scope="module")
def belgeler():
    return load_belgeler()


@pytest.fixture(scope="module")
def index(belgeler):
    return indeksle(chunkla_belgeler(belgeler, 120, 20), fake_embed)


# 0. Belgeler
def test_load_belgeler(belgeler):
    assert isinstance(belgeler, list) and len(belgeler) >= 4
    assert all(set(b.keys()) >= {"baslik", "metin"} for b in belgeler)
    basliklar = {b["baslik"] for b in belgeler}
    assert len(basliklar) == len(belgeler)  # başlıklar benzersiz


# 1. parcala — overlap'li bölme
def test_parcala_temel():
    metin = "a" * 250
    parcalar = parcala(metin, 100, 20)
    assert all(len(p) <= 100 for p in parcalar)
    assert len(parcalar) >= 3           # 250 kar / 80 adım → 4 parça
    # overlap: birleştirince orijinalden uzun (tekrar var)
    assert sum(len(p) for p in parcalar) > len(metin)


def test_parcala_kisa():
    assert parcala("kısa", 100, 20) == ["kısa"]


# 2. chunkla_belgeler — kaynak ekli
def test_chunkla_belgeler(belgeler):
    chunks = chunkla_belgeler(belgeler, 120, 20)
    assert len(chunks) >= len(belgeler)
    assert all(set(c.keys()) == {"metin", "kaynak"} for c in chunks)
    assert {c["kaynak"] for c in chunks} == {b["baslik"] for b in belgeler}


# 3. benzerlik — cosine
def test_benzerlik():
    assert abs(benzerlik([1, 0], [1, 0]) - 1.0) < 1e-6      # aynı
    assert abs(benzerlik([1, 0], [0, 1]) - 0.0) < 1e-6      # dik
    assert benzerlik([0, 0], [1, 1]) == 0.0                 # sıfır vektör
    assert benzerlik([1, 1], [2, 2]) > 0.99                 # aynı yön


# 4. indeksle — vektör ekli
def test_indeksle(index):
    assert all("vektor" in c for c in index)
    assert all(len(c["vektor"]) == len(VOCAB) for c in index)


# 5. getir — doğru chunk (anlamsal)
def test_getir_pil(index, belgeler):
    top = getir("pil ömrü ve şarj süresi ne kadar?", index, fake_embed, k=1)
    assert top[0]["kaynak"] == "Pil ve Şarj"


def test_getir_kamera(index):
    top = getir("kamera kaç megapiksel video çeker?", index, fake_embed, k=1)
    assert top[0]["kaynak"] == "Kamera"


def test_getir_k(index):
    assert len(getir("garanti", index, fake_embed, k=3)) == 3


# 6. sor — cevap + kaynak
def test_sor_kaynak(index):
    cevap, kaynaklar = sor("pil ömrü ne kadar?", index, fake_embed, fake_llm, k=2)
    assert isinstance(cevap, str) and len(cevap) > 0
    assert "Pil ve Şarj" in kaynaklar
    assert kaynaklar == sorted(set(kaynaklar))   # sıralı + benzersiz


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────
import requests


def _send_score(user_score):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return
    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return
    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {"user_id": USER_ID, "project_id": PROJECT_ID, "user_score": user_score, "is_auto": True}
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
