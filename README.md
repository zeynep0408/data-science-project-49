# DS-49 — Belge Asistanı (Temel RAG)

**Modül**: RAG ve Vektör Veritabanları (ML-08) • **Süre**: 2-3 saat

## 🎯 Proje Senaryosu

Elinde birkaç **belge** var (bir ürün kılavuzunun bölümleri: pil, kamera, garanti...). Kullanıcı doğal dille soruyor: *"Pil ömrü ne kadar?"*, *"Kamera kaç megapiksel?"*

LLM bu belgeleri bilmez — uydurabilir. Çözüm **RAG**: soruya en ilgili belge parçalarını **bul**, LLM'e **bağlam** olarak ver, cevabı **kaynağıyla** döndür.

Derste LangChain + Chroma bunları senin için yapıyordu. Bu projede aynı hattı **sıfırdan** kuruyorsun — böylece "hazır araç ne yapıyor" tam anlıyorsun: **parçala → embed → cosine ile getir → kaynaklı cevap.**

## 🧩 Kuracağın Hat

1. `parcala` / `chunkla_belgeler` — belgeleri üst üste binen (overlap) parçalara böl, her parçaya kaynak ekle
2. `benzerlik` — iki vektör arası **cosine** (elle)
3. `indeksle` — her parçayı embed et
4. `getir` — soruya en yakın top-k parçayı cosine ile bul
5. `sor` — getir → bağlamlı prompt → LLM → **cevap + kaynak**

## 🔌 embed ve llm Nasıl Çalışıyor? (önemli)

Fonksiyonların `embed` ve `llm`'i **parametre olarak alır** (dependency injection):

- **`embed`**: metin (str) → sayı listesi (vektör). Gerçekte `sentence-transformers` ya da OpenAI embedding.
- **`llm`**: prompt (str) → cevap (str). Gerçekte OpenAI.
- **Testlerde** deterministik sahte `embed`/`llm` verilir → **API key GEREKMEZ**, puan hep aynı.

Aynı kod, sadece verdiğin fonksiyonlar değişir — bu, test edilebilir RAG kodu yazmanın profesyonel yoludur.

## 📦 Kurulum

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python watch.py                 # otomatik test (dosya değişince)
# veya
pytest tests/test_question.py -v
```

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

`USER_ID`'ni Kaizu profilinden alıp yaz (`PROJECT_ID`'ye dokunma). Testler geçince skor otomatik gider.

## 🔑 API key? (opsiyonel)

Puanlama için **gerekmez** (mock). Gerçek embed/LLM ile denemek istersen `cp .env.example .env` + anahtarını koy (`.env` asla commit edilmez).
