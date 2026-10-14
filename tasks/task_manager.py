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
    adim = boyut - overlap
    return [metin[i:i + boyut] for i in range(0, len(metin), adim)]
    pass


def chunkla_belgeler(belgeler, boyut=120, overlap=20):
    return [
        {"metin": parca, "kaynak": doc["baslik"]}
        for doc in belgeler
        for parca in parcala(doc["metin"], boyut, overlap)
    ]
    pass


def benzerlik(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return dot / (norm_v1 * norm_v2)
    pass


def indeksle(chunks, embed):
    return [
        {**chunk, "vektor": embed(chunk["metin"])}
        for chunk in chunks
    ]
    pass


def getir(soru, index, embed, k=3):
    soru_vektor = embed(soru)
    skorlu_liste = []

    for item in index:
        skor = benzerlik(soru_vektor, item["vektor"])
        skorlu_liste.append((skor, item))

    skorlu_liste.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in skorlu_liste[:k]]
    pass


def sor(soru, index, embed, llm, k=3):
    ilgili_chunklar = getir(soru, index, embed, k=k)
    baglam = "\n".join([chunk["metin"] for chunk in ilgili_chunklar])
    prompt = f"Bağlam:\n{baglam}\n\nSoru: {soru}\nCevap:"
    cevap = llm(prompt)
    
    kaynaklar = []
    for chunk in ilgili_chunklar:
        kaynak = chunk["kaynak"]
        if kaynak not in kaynaklar:
            kaynaklar.append(kaynak)
            
    return cevap, kaynaklar
    pass
