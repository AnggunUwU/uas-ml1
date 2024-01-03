# Laporan Proyek Machine Learning
### Nama : Anggun Lisnawati
### Nim : 211351019
### Kelas : Malam B

## Domain Proyek

analisis segementasi pelanggan adalah salah satu teknik yang baik untuk d pakai dan dilakukan guna mengatur strategi penjualan kedepannya. dengan adanya analisis segmentasi pelanggan, penjual atau pebisnis dapat mebuat beberapa promo dan strategi bisnis lainnya untuk meningkatkan penjualan berdasarkan kebiasaan pelanggan yang sudah di kelompokan.

## Business Understanding

berdasarkan domain proyek yang sudah dijelaskan sebelumnya, maka perlu dibuat sistem yang mampu menganalisis dan memberikan informasi terkait pengelompokan pelanggan yang selanjutnya akan membantu pebisnis dalam melakukan treatment kepada pelanggan itu sendiri. proyek ini dibuat menggunakan metode clustering dengan model RFM dan algoritma K-means.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Di perlukannya segmentasi pelanggan untuk menentukan strategi bisnis
- pengelompokan kebiasaan antar pelanggan
### Goals

Menjelaskan tujuan dari pernyataan masalah:
- segmentasi pelanggan digunakan untuk mengetahui kelompok pelanggan berdasarkan kebiasaan belanja nya
- pembagian kelompok tersebut bermaksud untuk meningkatkan strategi bisnis yang memperngaruhi pendapatan

    ### Solution statements
    - pembuatan sistem yang mampu melakukan pengelompokan pelanggan menggunakan metode clustering dengan model RFM dan algoritma K-Means
    - metrik evaluasi yg d pakai adalah metode elbow

## Data Understanding

dataset Ini adalah kumpulan data transaksional yang berisi semua transaksi yang terjadi antara 01/12/2010 dan 09/12/2011 untuk ritel online non-toko yang berbasis di Inggris dan terdaftar. Banyak pelanggan perusahaan ini adalah pedagang grosir. dataset ini berukuran 541909 baris dan 8 kolom.

dataset: [E-Commerce Data](https://www.kaggle.com/datasets/carrie1/ecommerce-data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- InvoiceNo : Nomor pembelian
- StockCode : kode dari stock barang
- Description : nama barang
- Quantity : jumlah barang yang dibeli
- InvoiceDate : tanggal transaksi pembelian barang
- UnitPrice : harga barang
- CustomerID : ID pelanggan
- Country : negara tempat transaksi

tipe data:

<img width="236" alt="image" src="https://github.com/AnggunUwU/uas-ml1/assets/149172875/2374b31e-f56c-4f70-a015-5d35ee376c93">


## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

## Deployment
pada bagian ini anda memberikan link project yang diupload melalui streamlit share. boleh ditambahkan screen shoot halaman webnya.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

