import tensorflow as tf
from tensorflow.keras import layers, models

# 1. ŞANTİYE AYARLARI
# Klasörünün adını tam buraya giriyoruz
klasor_yolu = "veri_seti"

print("Veriler klasörlerden taranıyor ve şantiyeye taşınıyor...")

# 2. VERİ BORU HATTI (DATA PIPELINE)
# Modelin kendini test edebilmesi için verilerin %20'sini 'validation' (doğrulama) olarak ayırıyoruz.
# Resimleri otomatik olarak 28x28 boyutuna ve siyah-beyaz (grayscale) formata zorluyoruz.
train_ds = tf.keras.utils.image_dataset_from_directory(
    klasor_yolu,
    validation_split=0.2,
    subset="training",
    seed=123,
    color_mode="grayscale",
    image_size=(28, 28),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    klasor_yolu,
    validation_split=0.2,
    subset="validation",
    seed=123,
    color_mode="grayscale",
    image_size=(28, 28),
    batch_size=32
)

# ÇOK KRİTİK: Keras klasörleri hangi sırayla okudu? Bunu bilmemiz lazım!
sinif_isimleri = train_ds.class_names
print(f"\nDIKKAT! Modelin Öğrendiği Sınıf Sıralaması: {sinif_isimleri}\n")

# Pikselleri 0-255 aralığından 0-1 aralığına çekiyoruz (Modelin daha zeki olması için şart)
normalization_layer = layers.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

print("14 Sınıflı Yeni Nesil Hesap Makinesi Mimarisi İnşa Ediliyor...")
# 3. YENİ BEYİN MİMARİSİ
model = models.Sequential([
    # Göz Katmanı
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # Derin Odak Katmanı
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    # Nöron sayısını artırdık çünkü artık semboller de var, daha çok düşünmesi lazım
    layers.Dense(128, activation='relu'),

    # VE BÜYÜK DEĞİŞİM: ÇIKIŞ KATMANI ARTIK 14 İHTİMAL SUNUYOR!
    layers.Dense(14, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Eğitim Ateşleniyor! Arkana yaslan ve veri bilimi şovunu izle...")
# Veri setimiz karmaşıklaştığı için epochs'u (eğitim turu) 10 yaptık
model.fit(train_ds, validation_data=val_ds, epochs=10)

# 4. YENİ BEYNİ KAYDET
model.save('hesap_makinesi_beyni.h5')
print("\nMUAZZAM İŞ! Yeni 14 sınıflı model 'hesap_makinesi_beyni.h5' olarak başarıyla kaydedildi.")