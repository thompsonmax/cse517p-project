import matplotlib.pyplot as plt

unformatted_query_results = """English
608683950
French
189083656
German
93542410
Multilingual
46820028
Spanish
31475894
Latin
27044076
Unknown
20000494
Italian
19199889
Portugueuse
11711868
Greek
10621382
Polish
10226816
Portuguese
7815010
Russian
6287940
Swedish
3964456
Dutch
3784730
Danish
3756530
Czech
3702634
Hungarian
2952358
Bulgarian
2725993
Indonesian
2623665
Finnish
2569158
Romanian
2517783
Slovak
2390054
Estonian
2300739
Maltese
2246804
Japanese
1954166
Lithuanian
1903415
Ukrainian
1802097
Latvian
1581690
Chinese
1505306
Slovenian
1408892
Croatian
1397567
Arabic
1160155
Norwegian
890082
Haitian Creole (Latin script)
845447
null
775457
Cebuano
706791
Norwegian Nynorsk
703360
Turkish
615354
Korean
522917
Spanish; Castilian
494194
Catalan
476848
Hebrew
445253
Irish
405527
Basque
385209
Hindi
350372
Armenian
325659
Vietnamese
285349
Serbian
283419
Persian
261009
Occitan
240362
Egyptian Arabic
230038
Asturian
219360
Kinyarwanda
202723
Telugu
184650
Thai
173619
Welsh
172000
Azerbaijani
162482
Georgian
156896
Walloon
145726
Sardinian
144492
Tamil
141114
Javanese
136457
Kazakh
134511
Bengali
132169
Wolof
128501
Galician
128003
Kirghiz, Kyrgyz
126741
Greek, Modern
125311
Serbo-Croatian
117346
Amharic
113929
Tatar
111815
Chechen
109622
Esperanto
108060
Malayalam
106383
Khmer
103825
Traditional Chinese
101819
Urdu
98538
Burmese
92681
Kannada
88927
Yiddish
87706
Albanian
86693
Quechua
85044
Standard Latvian
84634
Bosnian
82402
Macedonian
79437
Malay
74675
Belarusian
71283
Icelandic
70910
Standard Malay (Latin script)
70711
Punjabi
70223
Uzbek
68151
Mon
65900
Papiamento
64555
Sinhala, Sinhalese
64427
Sundanese
61491
Waray
60436
Yoruba
57326
Marathi
56155
Somali
54546
Venetian
52765
Luxembourgish
50672
Simple English
50145
Afrikaans
41922
South Azerbaijani
39252
Xhosa
37877
Catalan; Valencian
36992
Slovene
36840
Romanian, Moldavan
35676
Nepali
35199
Iloko
34103
Zulu
33214
Sinhala
33039
Scottish Gaelic
31556
Kyrgyz
31278
Breton
30879
Nigerian Fulfulde
29942
West Frisian
28722
Uyghur
28328
Odia
28208
Northern Uzbek
26079
Gujarati
25357
Bashkir
25313
Friulian
25231
Shan
24214
Belarusian (Taraškievica)
23782
N'Ko
23521
Hausa
23478
Min Nan Chinese
23012
Sicilian
22781
Scots
22221
Tagalog
21881
Swahili
20124
Chuvash
19489
Japanese (Japanese script)
19395
Alemannic
19272
Bishnupriya Manipuri
18665
Aragonese
18384
Igbo
17187
Limburgish
16740
Minangkabau
15865
Goan Konkani
14400
Lingala
14350
Udmurt
14241
Central Kurdish
14145
Cantonese
14143
Tajik
13512
Kotava
13352
Western Armenian
13299
Sanskrit
13061
Lombard
11806
Malagasy
11792
Klingon
11490
Santali
11322
Newar
10436
Norwegian Bokmål
9372
Kurdish
9092
Ladin
8569
Mongolian
8547
Low German
8178
Shona
7967
Assamese
7752
Piedmontese
7540
Latgalian
6942
Swahil
6387
Angika
6346
Dutch Low Saxon
6339
Yakut
6244
Tibetan
6187
Sakizaya
6183
Upper Sorbian
5832
Corsican
5651
Maithili
5574
Silesian
5521
West Flemish
5018
Western Punjabi
4995
Haitian Creole
4989
JQ
4660
Lao
4639
Balinese
4473
Saraiki
4136
Wu Chinese
3994
Karachay-Balkar
3987
Faroese
3980
Kikuyu
3892
Madurese
3690
Bavarian
3664
Classical Chinese
3588
Navajo
3555
Dzongkha
3535
Romansh
3503
Manipuri
3356
Lower Sorbian
3208
Sindhi
3137
Mazanderani
3097
Kabyle
3011
Kabiye
2994
Guarani
2879
Crimean Tatar
2830
Russia Buriat
2772
Chichewa
2545
Volapük
2529
Zazaki
2397
Twi
2203
Ossetian
2120
Interlingue
2039
Ligurian
1867
Central Bikol
1786
Kashubian
1741
Gan Chinese
1731
North Frisian
1702
Pa'O
1508
Dagbani
1493
Swati
1465
Moksha
1460
Zeelandic
1387
Bambara
1379
Tuvan
1362
Ido
1334
Pashto
1312
Lezgian
1308
Ghanaian Pidgin English
1305
Moroccan Arabic
1290
Rusyn
1271
Aymara
1223
Bhojpuri
1177
Komi
1160
Amis
1128
Livvi-Karelian
1119
Hakka Chinese
1114
Pangasinan
1099
Norman
1092
Samogitian
1061
Kapampangan
1056
Gagauz
1041
Veps
1017
Church Slavonic
1002
Oromo
976
Tumbuka
919
Mirandese
907
Abkhaz
900
Kölsch
828
Luganda
800
Gothic
771
Manx
745
Gun
728
Buginese
674
Interlingua
644
Māori
625
Western Mari
620
Fiji Hindi
549
Neapolitan
528
Nahuatl
527
Talysh
522
Acehnese
518
Gorontalo
501
Ingush
499
Fon
494
Tarantino
485
Ladino
484
Inari Sami
461
Franco-Provençal
443
Min Dong Chinese
431
Erzya
430
Adyghe
417
Avaric
414
Mingrelian
392
Võro
386
Tongan
372
Pali
351
Banjar
349
Komi-Permyak
348
Cornish
322
Sranan Tongo
322
Emilian-Romagnol
319
Jamaican Patois
312
Gilaki
312
Northern Sami
305
Eastern Mari
301
Chavacano
301
Old English
301
Northern Sotho
283
Picard
281
Dhivehi
268
Lojban
255
Aramaic
226
Pontic
211
Awadhi
209
Taroko
194
Guianese Creole
190
Novial
165
Lak
155
Extremaduran
146
Saterland Frisian
144
Kashmiri
143
Turkmen
140
Tigrinya
117
Cherokee
104
Shilha
101
Nigerian Pidgin
97
Farefare
89
Hawaiian
84
Zhuang
82
Banyumasan
77
Wayuu
69
Inuktitut
67
Pennsylvania German
67
Ewe
54
Kongo
51
Venda
49
Chamorro
47
Fijian
38
Kimbundu
32
Tok Pisin
31
Bislama
31
Tswana
29
Pitcairn-Norfolk
24
Rundi
21
Kalmyk
7"""

if __name__ == "__main__":
    n = 25

    text = unformatted_query_results.splitlines()
    languages = [line for line in text if not line.isdigit()]
    tokens = [line for line in text if line.isdigit()]

    print('number of languages:', len(languages))

    if len(languages) != len(tokens):
        raise ValueError("count languages != count tokens")
    
    languages = languages[:n]
    tokens = tokens[:n]

    plt.bar(languages, tokens)
    plt.xlabel('Languages')
    plt.xticks(languages, languages, rotation=45)
    plt.ylabel('Tokens')
    plt.yscale('log')
    plt.title(f'Top {n} Languages by Tokens in Dataset')
    plt.show()


