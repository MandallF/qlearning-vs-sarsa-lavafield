# Lav Tarlası — Pekiştirmeli Öğrenme Projesi

**Ders:** Pekiştirmeli Öğrenmeye Giriş — Bursa Teknik Üniversitesi
**Öğrenci:** Kubilay İnanç (22360859047) — Bilgisayar Mühendisliği
**Öğretim Üyesi:** Dr. İlhan Tunç

## Özet

Bu projede, lav şeridiyle bölünmüş özel bir **ızgara dünyası (GridWorld)**
ortamı tasarladım ve ajanı **Q-Learning** ve **SARSA**
tablolu pekiştirmeli öğrenme yöntemleriyle eğittim. Ortam bir Markov Karar
Süreci (MKS) olarak modellenmiştir. Amaç, ajanın başlangıçtan ödül bölgesine
(hedef) güvenli bir rota öğrenmesidir.

Ortam, **riskli ama kısa** (lavın hemen üstünden) ile **güvenli ama uzun**
yol arasındaki tercihi açığa çıkaracak şekilde tasarlanmıştır. Böylece
Q-Learning (off-policy) ve SARSA (on-policy) arasındaki davranış farkı
gözlemlenir. Karşılaştırma için ayrıca **rastgele politika** (alt referans)
ve **BFS** ile bulunan en kısa güvenli yol (model-tabanlı üst referans)
kullanılmıştır.

## Kurulum

```bash
py -m pip install -r requirements.txt
```

## Çalıştırma

```bash
py main.py
```

Program; eğitimi yapar, sayısal özeti ekrana yazar (`../sonuclar.txt`
dosyasına da kaydeder) ve tüm grafikleri `../figurler/` klasörüne PNG olarak
üretir.

## Dosyalar

| Dosya             | Açıklama                                                        |
|-------------------|-----------------------------------------------------------------|
| `environment.py`  | `LavaGridWorld` — ortam (MDP: durum, eylem, ödül, geçiş)         |
| `agents.py`       | `TDAgent` — Q-Learning / SARSA + epsilon-greedy |
| `baselines.py`    | Rastgele politika ve BFS en kısa güvenli yol                    |
| `experiments.py`  | Eğitim döngüsü, çok-tohumlu deney, açgözlü değerlendirme         |
| `visualize.py`    | Tüm grafiklerin üretimi (matplotlib)                            |
| `main.py`         | Her şeyi çalıştıran ana dosya                                   |

## Üretilen Grafikler

1. `01_ortam.png` — ortam şeması
2. `02_ogrenme_egrileri.png` — öğrenme eğrileri (çevrimiçi ödül)
3. `03_adim_sayisi.png` — epizod başına adım sayısı
4. `04_ogrenilen_rotalar.png` — öğrenilen açgözlü politikalar/rotalar
5. `05_karsilastirma.png` — yöntem karşılaştırması (açgözlü ödül)
