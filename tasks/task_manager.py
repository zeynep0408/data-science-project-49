"""
DS-49 — Belge Asistanı (Temel RAG)

Senaryo: Elinde birkaç belge var (bir ürün kılavuzu bölümleri). Kullanıcı doğal
dille soruyor ("pil ömrü ne kadar?"). Sen RAG hattını SIFIRDAN kuruyorsun:
belgeleri parçala → embed et → soruya en yakın parçaları getir → LLM'e bağlam
olarak ver → cevap + KAYNAK. Derste LangChain/Chroma bunları senin için yapıyordu;
burada içlerinde ne olduğunu elle kurup anlıyorsun.

⚠️ `embed` ve `llm` DIŞARIDAN parametre gelir (dependency injection):
  - Gerçek kullanımda kendi modelini verirsin (README'ye bak — sentence-transformers / OpenAI).
  - Testlerde deterministik sahte fonksiyonlar verilir → API key GEREKMEZ.
  `embed`: metin (str) → sayı listesi (vektör). `llm`: prompt (str) → cevap (str).

Her fonksiyonun `pass` kısmını doldur. Test: `python watch.py` veya
`pytest tests/test_question.py -v`
"""
import math


def load_belgeler():
    """Belgeler HAZIR verildi (RAG corpus'un). Bunu değiştirme;
    aşağıdaki fonksiyonları doldurman yeterli."""
    return [
        {"baslik": "Pil ve Şarj", "metin": "Nova X1 telefonun pili 4500 mAh kapasitelidir. Tam şarj yaklaşık 90 dakika sürer. Tipik kullanımda pil ömrü 1.5 gündür. Hızlı şarj desteklenir."},
        {"baslik": "Kamera", "metin": "Nova X1 arka kamerası 50 MP çözünürlüktedir. Gece modu ve portre modu bulunur. Video 4K 60fps kaydeder. Ön kamera 16 MP."},
        {"baslik": "Garanti", "metin": "Nova X1 için 2 yıl resmi garanti sunulur. Garanti üretim hatalarını kapsar; fiziksel hasar kapsam dışıdır. Onarım yetkili serviste yapılır."},
        {"baslik": "Kutu İçeriği", "metin": "Kutuda Nova X1 telefon, USB-C şarj kablosu, 33W adaptör, SIM çıkarma iğnesi ve kullanım kılavuzu bulunur."},
        {"baslik": "Su Direnci", "metin": "Nova X1 IP68 sertifikalıdır. 1.5 metre derinlikte 30 dakika suya dayanır. Tuzlu su ve kimyasallara karşı garanti verilmez."},
    ]


def parcala(metin, boyut=120, overlap=20):
    """
    Metni karakter bazlı, ÜST ÜSTE BİNEN (overlap) parçalara böl.
    Her adımda `boyut` karakter al, sonra `boyut - overlap` kadar ilerle.

    Args:
        metin: str
        boyut: int — parça uzunluğu (karakter)
        overlap: int — komşu parçaların paylaştığı karakter
    Returns:
        list[str]
    """
    pass


def chunkla_belgeler(belgeler, boyut=120, overlap=20):
    """
    Her belgeyi `parcala` ile böl; her parçaya belgenin başlığını KAYNAK olarak ekle.
    Döndürülen her eleman: {"metin": str, "kaynak": str}

    Args:
        belgeler: list[dict] — load_belgeler() çıktısı
    Returns:
        list[dict]
    """
    pass


def benzerlik(v1, v2):
    """
    İki vektör arasındaki COSINE benzerliğini hesapla:
        dot(v1, v2) / (norm(v1) * norm(v2))
    Sıfır vektörde 0.0 döndür (bölme hatası olmasın).

    Returns:
        float
    """
    pass


def indeksle(chunks, embed):
    """
    Her chunk'ı embed et; chunk'a "vektor" alanı ekleyip döndür.
    Her eleman: {"metin": ..., "kaynak": ..., "vektor": embed(metin)}

    Args:
        chunks: list[dict] — chunkla_belgeler çıktısı
        embed: callable — str → list[float]
    Returns:
        list[dict]
    """
    pass


def getir(soru, index, embed, k=3):
    """
    Soruyu embed et, index'teki her chunk ile COSINE benzerliğini hesapla,
    en yüksek benzerlikli ilk `k` chunk'ı döndür (vektörleriyle birlikte).

    Args:
        soru: str
        index: list[dict] — indeksle çıktısı
        embed: callable
        k: int
    Returns:
        list[dict] — en benzer k chunk
    """
    pass


def sor(soru, index, embed, llm, k=3):
    """
    Tam RAG cevabı:
      1. getir(soru, index, embed, k) → ilgili chunk'lar
      2. bağlam = chunk metinlerini birleştir
      3. cevap = llm(bağlamlı prompt)
      4. kaynaklar = getirilen chunk'ların benzersiz "kaynak"ları (sıralı)
    (cevap, kaynaklar) ikilisini döndür.

    Returns:
        tuple(str, list[str])
    """
    pass
