# Tiefenrecherche & Architektur-Kompendium: Multimodale KI-gestützte Schlafanalyse und Schlafapnoe-Detektion (Teamprojekt AIN7)

## Inhaltsverzeichnis
1. **Projektkontext, Prüfungs-Constraints & Forschungsprofil der Betreuer (HTWG Konstanz)**
2. **Detaillierte Einzelanalyse der 6 Kern-Paper & 10 ergänzender State-of-the-Art-Paper (2021–2026)**
3. **Vollständige 18-Spalten-Literatur-Matrix (alle 16 Referenzwerke komplett integriert)**
4. **Biophysik, Sensorik & Signalgewinnung in der Schlafmedizin (EEG, EOG, EMG, EKG/HRV/EDR, SpO2/PPG, Atemfluss, RIP/EIP, Akustik, Aktigraphie)**
5. **Öffentliche Datensätze & 6-Stufen-Pipeline zur Cross-Dataset-Harmonisierung**
6. **Konzept & Architektur der lokalen, DSGVO-konformen Ziel-Software (inkl. Explainable AI)**
7. **Arbeitsverteilung für 6 Personen (Work Packages 1–6) & Wochenzeitplan bis Ende Dezember 2026**

---

## 1. Projektkontext, Prüfungs-Constraints & Forschungsprofil der Betreuer

### 1.1 Offizielle Leitfrage & Projektziel (aus der Kick-off-Präsentation `Team_project_AIN_26WS`)
* **Leitfrage:** *„How can we leverage multimodal data to boost sleep disorder classification using machine learning?“*
* **Offizielles Ziel:** Entwicklung einer multimodalen Machine-Learning-/Deep-Learning-Pipeline in Python unter Nutzung öffentlich verfügbarer Datensätze zur Klassifikation von Schlafstörungen (insbesondere **Schlafapnoe / Hypopnoe**, optional gekoppelt mit **Schlafstadien-Klassifikation**).
* **Vorgegebene 5 Projektphasen laut Präsentation:**
  1. *Basics of AI:* Grundprinzipien (Rule-Based Systems vs. Classic Machine Learning mit handdesigned Features vs. Representation Learning / Deep Learning nach Goodfellow et al., 2016).
  2. *Basic Knowledge on Sleep, Sleep Apnea & Multimodality:* Physiologie des Schlafs, Polysomnographie (PSG) als klinischer Goldstandard, Schlafapnoe-Subtypen und multimodale Sensorik.
  3. *State-of-the-Art Review:* Systematischer Vergleich bestehender KI-Ansätze für Schlafapnoe und Schlafstadien-Scoring mit multimodalen Daten.
  4. *Development of ML Pipeline:* Implementierung der eigenen multimodalen Pipeline in Python auf öffentlichen Datensätzen.
  5. *Analysis & Critical Evaluation:* Performance-Evaluation, Vergleich mit der Literatur, Stärken/Limitationen und klinische Erklärbarkeit (Explainable AI).

### 1.2 Forschungsprofil eurer Betreuer am Ubiquitous Computing Lab (UC Lab, HTWG Konstanz)
Ein entscheidender Erfolgsfaktor ist das Verständnis, woran eure drei Betreuer konkret forschen:
* **Prof. Dr. Ralf Seepold** (`ralf.seepold@htwg-konstanz.de`, Leiter UC Lab & Interdisziplinäres Schlaflabor): Co-Autor von zwei eurer sechs Kern-Paper (**Gaiduk et al., 2023** und **Alarcón et al., 2023**). Seine Forschung (u. a. im *MORPHEUS*-Projekt der Carl-Zeiss-Stiftung und im ZIM-Projekt *„Sleep Lab at Home“ / SLaH*) zielt darauf ab, die teure, unbequeme stationäre **Polysomnographie (PSG)** durch **nicht-invasive Portable Monitors (PMs)** für zu Hause zu ergänzen. Er legt größten Wert auf **Explainable AI (Grad-CAM)** für Schlafmediziner sowie **Inter-Dataset-Generalisierung** (Training auf Datensatz A, blinder Test auf Datensatz B und C).
* **Dr. Giulia Maria Mattia** (`gmattia@htwg-konstanz.de`): Spezialistin für multimodales Deep Learning in der Neurologie und Schlafmedizin. Ihre in der Präsentation zitierte Dissertation (*„Intelligence artificielle et maladies neurologiques : aider le diagnostic et améliorer la compréhension du comportement des réseaux de neurones convolutifs“*, 2022) untersucht gezielt **multimodale Datenfusion** und die **Interpretierbarkeit (Verständnis des internen Verhaltens von CNNs)** bei klinischen Entscheidungen.
* **MSc Daniel Vélez Gutiérrez** (`jvelezg@htwg-konstanz.de`): Forscht am UC Lab an nicht-invasiver Biosignal-Akquisition, tragbaren/ambienten Sensoren, Datenmodellierung und eHealth-Pipelines.

> **STRATEGISCHER HEBEL FÜR EUER TEAM (Direkter Anschluss an Alarcón et al., 2023):**
> Das Paper **Alarcón, Madrid, Seepold & Ortega (2023)** ist die unmittelbare Vorarbeit eures Lehrstuhls. Auf Seite 17 („Discussion“) benennen Prof. Seepold und Ko-Autoren explizit die **offenen Limitationen ihrer Arbeit**, die in künftigen Projekten gelöst werden sollen:
> 1. **AHI-Überschätzung durch fehlende Schlafzeit-Erkennung:** Sie teilen die Anzahl erkannter Apnoe-Fenster durch die *Total Recording Time (TRT, 8 Stunden)* statt durch die echte *Total Sleep Time (TST)*, weil ihr Modell Wachphasen nicht von Schlafphasen unterscheidet.
> 2. **Nur binäre Klassifikation (Apnoe vs. Nicht-Apnoe):** Keine Unterscheidung zwischen *Obstruktiver Apnoe (OSA)*, *Zentraler Apnoe (CSA)* und *Hypopnoe*.
> 3. **Informationsverlust durch 1-Hz-Downsampling:** Alle Signale wurden auf `1 Hz` heruntergerechnet, wodurch feine Kurvenformen der Atmungs-, EKG- und EEG-Signale verloren gehen.
> 4. **Fenster-Grenzen-Problem (60s-Windowing) & Z-Score-Verzerrung:** Apnoe-Events an Fenstergrenzen werden zerschnitten, und die fensterweise Z-Score-Normierung bläst minimale SpO2-Schwankungen (`< 3 %`) bei Gesunden künstlich zu Fehlalarmen auf.
>
> **Wenn euer 6er-Team genau diese vier Punkte löst (z. B. durch ein gekoppeltes System aus Schlafstadien-/Schlaf-Wach-Erkennung für exakte TST + multimodaler Multi-Klassen-Apnoe-Detektion + robuster Normalisierung + lokaler XAI-GUI), liefert ihr die exakte Weiterentwicklung der Lehrstuhl-Forschung!**

---

### 1.3 Prüfungsleistungen & Formale Vorgaben (aus `Team_project_Evaluation`)
Jedes der 6 Teammitglieder erhält eine **individuelle Note (Skala 1 bis 5)**, basierend auf:

| Teilprüfung | Gewichtung | Formale Anforderungen & Constraints | Kritische Konsequenz für euer 6er-Team |
| :--- | :--- | :--- | :--- |
| **Wissenschaftliches Paper** | **60 %** | • **Nur 1 gemeinsames Paper pro Gruppe**<br>• Format: **KES Conference Paper Template** (Elsevier Procedia Computer Science / Springer KES)<br>• Seitenumfang: **Minimum 8, Maximum 10 Seiten** (exkl. Literaturverzeichnis; max. ca. 7.000–8.000 Wörter)<br>• **Maximal 5 grafische Elemente insgesamt (Tabellen UND Abbildungen zusammen!)** | **Striktes 5-Elemente-Limit beachten:** Ihr dürft im gesamten Paper zusammen nur höchstens 5 Tabellen/Abbildungen verwenden! Plant daher **kombinierte Multi-Panel-Abbildungen** (siehe konkreten 5-Elemente-Plan in Abschnitt 7.3 dieses Berichts). |
| **Mündliche Präsentation** | **40 %** | • **1 gemeinsame Gruppenpräsentation**<br>• **15 Minuten Redezeit pro Teammitglied** (= **90 Minuten Gesamtvortrag** bei 6 Personen!) | Da jede Person **15 Minuten lang** einen eigenen Bereich tiefgehend präsentieren muss, braucht jedes der 6 Teammitglieder ein klar abgegrenztes **eigenes Arbeitspaket (Work Package)** mit eigenen Experimenten, eigenen Metriken oder eigener Systemkomponente. |

### 1.4 Empfohlene Werkzeuge (aus `Team_project_List_of_useful_tools`)
* **Literaturverwaltung:** **Zotero** (Empfehlung: gemeinsamer Zotero-Gruppenordner für alle 6 Mitglieder mit Word-/LaTeX-Plugin) oder **Mendeley**.
* **Literatur-Suchmaschinen:** Scopus, Google Scholar, IEEE Xplore / IEEE Access, PubMed.
* **Sprach- & Schreibwerkzeuge:** **Academic Phrasebank** (`https://www.phrasebank.manchester.ac.uk/`, Standardformulierungen für *Introduction, Methods, Results, Discussion*), **DeepL** und **Grammarly**.
* **Mindmaps:** **Xmind** zur schematischen Zusammenfassung von Review-Papern.
* **Online-ML-Kurse (falls Teammitglieder Grundlagen auffrischen wollen):** FUN-MOOC (`scikit-learn`), Harvard edX (*Using Python for Research* / *ML and AI with Python*), Andrew Ng Deep Learning Specialization (Coursera/YouTube).

---

## 2. Tiefenanalyse & Vergleich der 6 Kern-Paper + SOTA-Literatur (2021–2026)

### 2.1 Detaillierte Einzelanalyse der 6 bereitgestellten Paper

#### Paper 1: Alarcón, Madrid, Seepold & Ortega (2023)
* **Titel:** *Obstructive sleep apnea event detection using explainable deep learning models for a portable monitor*
* **Journal & Link:** *Frontiers in Neuroscience*, Vol. 17, 1155900 (2023) — `https://doi.org/10.3389/fnins.2023.1155900`
* **Zielsetzung:** Entwicklung eines erklärbaren Deep-Learning-Modells zur Detektion obstruktiver Schlafapnoe-Events (OSA) und zur Berechnung des **Apnoe-Hypopnoe-Index (AHI)** für den Einsatz auf einem tragbaren Monitor (PM).
* **Verwendete Signale (4 Kanäle):**
  1. Sauerstoffsättigung (`SpO2`, original `1 Hz`)
  2. Herzfrequenz (`HR`, original `1 Hz`)
  3. Thorakale Atmungsanstrengung (`Thor-Res`, RIP-Brustgurt, original `10 Hz` in SHHS bzw. `32 Hz` in MESA -> **downsampled auf `1 Hz`**)
  4. Abdominale Atmungsanstrengung (`Abdo-Res`, RIP-Bauchgurt, original `10 Hz` bzw. `32 Hz` -> **downsampled auf `1 Hz`**)
  *Hintergrund:* Das geplante Hardware-PM des UC Lab misst `SpO2`, `HR` und elektrische Impedanzpneumographie (`EIP`). Da öffentliche PSG-Datenbanken kein `EIP` enthalten, dienen `Thor-Res` und `Abdo-Res` als engstes physiologisches Äquivalent.
* **Datensätze & Splits (NSRR Repository):**
  * **SHHS2 (Training, Validation & interner Test):** `1.224` Patienten für Training (`70 %`), `244` für Validation (`20 %`), `163` für Test (`10 %`). Auswahlkriterien: Aufnahmedauer `>= 300 Min.`, wenig Artefakte/Missing Values, ausgewogenes Geschlecht (`44 %` Männer, `56 %` Frauen).
  * **SHHS1 (Externer Generalisierungs-Test):** `1.859` Patienten (`1.518` mit Apnoe, `341` ohne).
  * **MESA (Externer Generalisierungs-Test):** `976` Patienten (`863` mit Apnoe, `113` ohne).
* **Preprocessing & 4 untersuchte Modell-Varianten:**
  * **Windowing:** Nicht-überlappende **60-Sekunden-Fenster** (`60` Datenpunkte pro Kanal bei `1 Hz`, `480` Fenster pro `8h = 28.800s` Aufnahme). Ein Fenster erhält das Label `1` (Apnoe/Hypopnoe), wenn mindestens `10 s` eines annotierten Events darin liegen.
  * **Klassen-Balancierung:** `50 %` Apnoe-Fenster / `50 %` zufällig gezogene Nicht-Apnoe-Fenster (`316.316` Trainingsfenster, `61.890` Validierungsfenster, `42.836` Testfenster in SHHS2). Zusätzlich Test auf unbalancierten Daten (Verhältnis `1:3`).
  * **Artefakt-Entfernung:** Werte mit `SpO2 < 80 %` oder `> 100 %` sowie `HR < 40 bpm` oder `> 200 bpm` (inkl. Null-Dropouts) werden durch **lineare Interpolation** ersetzt.
  * **Normalisierung:** **Fensterweise Z-Score-Standardisierung** (`z = (X - μ) / σ` separat für jedes einzelne 60s-Fenster berechnet).
  * **Die 4 Trainings-Setups:**
    * *Model 1 (`TrainDat1`):* Ohne Artefaktentfernung, ohne Z-Score.
    * *Model 2 (`TrainDat2`):* Mit Artefaktentfernung, ohne Z-Score.
    * *Model 3 (`TrainDat3`):* Ohne Artefaktentfernung, mit Z-Score.
    * *Model 4 (`TrainDat4`):* **Mit Artefaktentfernung UND Z-Score (Bestes Modell!)**.
* **Modellarchitektur & Hyperparameter (Model 4):**
  * **1D-CNN mit Global Average Pooling (GAP):** 6 Convolutional Layers (`Conv1D` + `BatchNormalization` + `ReLU` + `Dropout`), gefolgt von einem `GAP`-Layer (erhält die zeitliche Lokalisierung für Grad-CAM!) und einem `Dense(1, activation='sigmoid')` Output-Layer.
  * Optimiert mit **Keras Tuner (Hyperband, 90 Trials, Batch Size = 1.024, Adam, Learning Rate = 0.001, Binary Crossentropy)**:
    * Layer 1: `128` Filter, Kernel Size `7`, Dropout `0.3`
    * Layer 2: `192` Filter, Kernel Size `5`, Dropout `0.3`
    * Layer 3: `224` Filter, Kernel Size `3`, Dropout `0.4`
    * Layer 4: `96` Filter, Kernel Size `7`, Dropout `0.2`
    * Layer 5: `256` Filter, Kernel Size `9`, Dropout `0.3`
    * Layer 6: `96` Filter, Kernel Size `9`, Dropout `0.5`
* **Quantitative Ergebnisse:**
  * **Per-Segment (60s-Fenster, balanciert, Model 4):**
    * **SHHS2 Test:** Accuracy **84,3 %**, Sensitivity **82,5 %**, Specificity **86,0 %**, Precision **85,5 %**, **AUC 92,1 %** (Positive Likelihood Ratio `LR+ = 5,89`, erfüllt die klinische PM-Vorgabe von Collop et al., 2011: Sensitivity `>= 82,5 %` und `LR+ >= 5`).
    * **MESA Test (extern):** Accuracy **74,5 %**, Sensitivity **76,0 %**, Specificity **72,8 %**, **AUC 80,8 %**.
    * **SHHS1 Test (extern):** Accuracy **70,0 %**, Sensitivity **53,7 %**, Specificity **86,3 %**, **AUC 77,7 %**.
    * *Einfluss des Preprocessings:* Z-Score (*Model 3/4* vs. *Model 1/2*) steigerte die Accuracy auf SHHS2 massiv von `72,3 %` auf `84,3 %`. Die Artefaktentfernung verbesserte vor allem den stark artefaktbelasteten externen Datensatz SHHS1 (Sensitivity stieg von `46,0 %` bei Model 3 auf `53,7 %` bei Model 4).
  * **Per-Patient (AHI-Schätzung & Schweregrad):**
    * Berechnungsformel: `AHI_pred = (Anzahl als OSA erkannter 60s-Fenster) / Total Recording Time (8h)`.
    * Bestimmtheitsmaß `R²`: **0,65** (SHHS2), **0,64** (SHHS1), **0,62** (MESA).
    * Bei Schweregrad-Schwellen (`AHI >= 5`, `>= 15`, `>= 30`): Für schwere Apnoe (`AHI >= 30`) sehr hohe Treffsicherheit (Accuracy **91,4 %** auf SHHS2, **92,3 %** auf SHHS1, **88,4 %** auf MESA), aber starke Überschätzung bei Gesunden (`AHI < 5`: Spezifität nur `7–16 %`).
  * **Explainable AI (Grad-CAM):**
    * Zeigt direkt auf den 4 Zeitreihenkurven farblich an, welche Sekunden des 60s-Fensters für die Klassifikation entscheidend waren.
    * Das Modell achtet primär auf die **Kombination aus SpO2-Desaturierung und gegenläufigem Herzfrequenz-Anstieg (Bradykardie -> Tachykardie-Arousal)** bei gleichzeitig flacher `Thor-Res`-/`Abdo-Res`-Kurve.

---

#### Paper 2: Peng, Sun, Zhou & Zhang (2025)
* **Titel:** *AI-driven approaches for automatic detection of sleep apnea/hypopnea based on human physiological signals: a review*
* **Journal & Link:** *Health Information Science and Systems*, Vol. 13, Issue 7, Springer (2025) — `https://doi.org/10.1007/s13755-024-00320-8`
* **Fokus:** Umfassendes Review über die 5-stufige KI-Pipeline (*Data Acquisition -> Pre-processing -> Model Construction -> Classification -> Evaluation*) für Schlafapnoe/Hypopnoe, getrennt nach den **5 Haupt-Modalitäten**: (1) **Nasal Airflow**, (2) **SpO2**, (3) **EKG**, (4) **EEG** und (5) **Snoring Sound (Schnarchgeräusche)**.
* **Kernaussagen & Techniken:**
  * **1D-Rohsignale vs. 2D-Zeit-Frequenz-Bilder:** Viele Studien wandeln 1D-Signale vor der Klassifikation in 2D-Bilder um, damit 2D-CNNs (ResNet, Inception-v3, VGG19, ShuffleNet) gleichzeitig Zeit- und Frequenzmuster erfassen:
    1. **Continuous Wavelet Transform (CWT):** Vergleicht das Signal `f(t)` mit skalierten (`a`) und verschobenen (`b`) Mutter-Wavelets `ψ(t)` (z. B. analytisches Morlet-Wavelet) und erzeugt ein **Skalogramm** mit hoher Zeitauflösung bei hohen Frequenzen und hoher Frequenzauflösung bei tiefen Frequenzen.
    2. **Short-Time Fourier Transform (STFT):** Berechnet die Fourier-Transformation über ein gleitendes Zeitfenster `h(τ - t)` fester Breite und erzeugt ein **Spektrogramm**.
    3. **Gabor Transform (GT) / Smoothed Gabor Spectrogram (SGS):** Nutzt ein Gauß-Fenster `g(t) = exp(-π * t²)` vor der Fourier-Transformation für optimale Zeit-Frequenz-Lokalisierung (z. B. *OSACN-Net* auf EKG mit **94,81 %** Accuracy).
    4. **Recurrence Plots (RP):** Visualisieren wiederkehrende Zustände dynamischer Systeme im Phasenraum als 2D-Matrix (kombiniert für `EEG + EKG + Airflow` mit ResNet-18/ShuffleNet: **90,72 %** auf MIT-BIH und **90,45 %** auf UCDDB).
    5. **Mel-Spektrogramme & MFCCs:** Standard für Schnarch-Audio (z. B. VGG19+LSTM erreicht **99,31 %** für Normal- vs. OSA-Schnarchen).
  * **Überlegenheit multimodaler Fusion:**
    * *Nasal Airflow + SpO2 (Peng et al., 2024):* Dual-Branch 1D-CNN mit **Efficient Channel Attention (ECA)** auf 10s-Segmenten erreicht **95,91 %** Accuracy für 3 Klassen (Normal / Apnoe / Hypopnoe).
    * *SpO2 + EKG/RR-Intervalle (Paul et al., 2022):* Ein Feedforward Neural Network (FNN) verbessert sich von `80,04 %` (nur EKG) bzw. `90,78 %` (nur SpO2) auf **91,83 %** bei Kombination beider Signale; eine SVM auf R-R-Intervallen + SpO2-Statistiken erreicht **99,1 %**.
    * *Snoring Sound + Tracheal Motion + SpO2 (Montazeri et al., 2023):* Dual-DNN erreicht **92,45 %** für `AHI >= 15`.
  * **Identifizierte Haupt-Herausforderungen:** (1) Starke **Class Imbalance** (insbesondere sind Hypopnoen deutlich seltener und schwerer zu erkennen als Apnoen -> Lösung: Adaptive Sliding Window Weighting, Focal Loss, SMOTE/GANs), (2) **Multi-Center Domain Shift** (ein 1D-CNN auf SpO2 erreichte `72,8 %` auf CHAT, fiel aber auf `60,2 %` bei UofC und `60,0 %` bei BUH!).

---

#### Paper 3: Fathima & Ahmed (2025)
* **Titel:** *Sleep Apnea Detection Using EEG: A Systematic Review of Datasets, Methods, Challenges, and Future Directions*
* **Journal & Link:** *Annals of Biomedical Engineering*, Vol. 53, Issue 5, pp. 1043–1067, Springer (2025) — `https://doi.org/10.1007/s10439-025-03691-5`
* **Fokus:** Systematisches PRISMA-/PICO-Review über **63 Studien (2010–2024)** zur Schlafapnoe-Detektion und Subtypisierung (OSA, CSA, MSA) über **EEG-Signale**.
* **Schlüsselerkenntnisse für euer Projekt:**
  * **Neurophysiologischer Mechanismus:** Atemstillstand -> O2-Abfall (Hypoxie) & CO2-Anstieg (Hyperkapnie) -> Chemorezeptoren aktivieren Hirnstamm (*Medulla oblongata, Pons*) und *Amygdala* -> **kortikales Arousal (Weckreaktion)** im EEG, Unterdrückung von Schlafspindeln und K-Komplexen im N2-Schlaf sowie charakteristische Energie-Verschiebungen zwischen den EEG-Frequenzbändern.
  * **Morphologische Unterschiede im EEG (UCDDB-Analyse):**
    * *Normale Atmung:* Gleichmäßige, rhythmische Hintergrundaktivität ohne abrupte Amplitudensprünge.
    * *Obstruktive Schlafapnoe (OSA):* Phasen deutlich unterdrückter Amplitude während der Obstruktion, gefolgt von **scharfen hochamplitudigen Spikes (Arousal)** beim Luftschnappen.
    * *Zentrale Schlafapnoe (CSA):* Weichere, glatte Amplitudensuppression und -erholung (fehlendes zentrales Atemkommando).
    * *Gemischte Schlafapnoe (MSA):* Kombination aus anfänglicher glatter CSA-Suppression und anschließendem scharfem OSA-Arousal-Spike.
  * **Kanalwahl & Rechts-Hemisphären-Dominanz (`C4-A1` vs. `C3-A2`):**
    * Die zentralen Ableitungen **C3** und **C4** sind die mit Abstand am häufigsten genutzten Kanäle (in 59 von 63 Studien reicht bereits 1 EEG-Kanal!).
    * **Wichtiger Befund:** Die **rechte zentrale Ableitung (`C4-A1` bzw. `C4-M1`)** erzielt eine signifikant bessere Klassifikationsleistung (**92,73 %** mit BiLSTM) als die linke (`C3-A2`, **90,05 %**), da die **rechte Großhirnhemisphäre im Schlaf bei Apnoe-Patienten stärker auf respiratorische Stressreize reagiert** als die linke! Werden beide symmetrischen Kanäle (`C3` + `C4`) kombiniert, steigt die Genauigkeit weiter an.
  * **Sub-Frame-Segmentierung für Stationarität:**
    * Ein 30s- oder 60s-EEG-Fenster ist **nicht-stationär**, weil sich das Signalregime beim Übergang von Apnoe zu Arousal innerhalb weniger Sekunden ändert.
    * Lösung: Unterteilung des 30s-/60s-Fensters in überlappende **Sub-Frames von `1 s`, `2 s`, `3 s`, `4 s` oder `10 s`** (Stationarität prüfbar über den *Augmented Dickey-Fuller Test*). Das isoliert lokale Arousal-Spitzen und verhindert, dass kurze 10s-Apnoen im 60s-Fenster-Mittelwert verschwinden.
  * **Stärkste EEG-Features für Schlafapnoe:**
    * *Delta-Band-Power-Ratio (`0,5–4 Hz`):* Fällt zu Beginn einer Apnoe abrupt ab, bleibt während des Events niedrig und steigt nach Ende der Apnoe wieder an.
    * *Beta-Band Energy Transients (`13–30 Hz`):* Zeigt beim Arousal am Ende der Apnoe einen scharfen Energie-Peak.
    * *Nichtlineare Komplexitätsmaße:* **Fuzzy Entropy** (in Delta bis Alpha während Apnoe niedriger, in Sigma `12–16 Hz` und Beta höher als normal), **Lempel-Ziv Complexity** (in Apnoe-Subbändern erhöht, `96 %` Acc.), **Detrended Fluctuation Analysis (DFA)** (`95,1 %` Acc.) sowie reduzierte interhemisphärische **Mutual Information (MI)**.

---

#### Paper 4: Gaiduk, Alarcón, Seepold & Madrid (2023)
* **Titel:** *Current status and prospects of automatic sleep stages scoring: Review*
* **Journal & Link:** *Biomedical Engineering Letters*, Vol. 13, pp. 247–272, Springer (2023) — `https://doi.org/10.1007/s13534-023-00299-3`
* **Fokus:** Semi-systematisches Review über **125 Journal-Paper (2018–2022)** zur vollautomatischen Klassifikation der 5 Schlafstadien nach **AASM-Standard** (**Wake [W], N1, N2, N3, REM [R]** in 30s-Epochen).
* **Kernaussagen & Benchmarks:**
  * **Mensch vs. KI:** Selbst erfahrene Schlafmediziner erreichen beim manuellen Scoring derselben PSG nur eine Übereinstimmung (**Inter-Rater Reliability**) von **Cohen's Kappa `κ ≈ 0,76`** (Spanne `0,71–0,81`). Moderne Deep-Learning-Modelle auf `EEG` oder `EEG + EOG + EMG` erreichen bereits **Accuracy `> 90 %`** und **Cohen's Kappa `κ > 0,80`** (F1-Score `85–90 %`).
  * **Leistungseinbruch bei Schlafapnoe-Patienten:** Werden Schlafstadien-Modelle nur auf gesunden Probanden trainiert, sinkt ihre Genauigkeit auf Patienten mit Schlafapnoe drastisch (z. B. Kappa fällt von `0,79` auf `0,66`), weil Apnoe-Arousals die Schlafarchitektur fragmentieren. Daher muss das Schlafstadien-Modul zwingend auf **gemischten Kohorten (Gesunde + Apnoe-Patienten)** trainiert werden!
  * **Alternative / Komfortable Signale:** Schlafphasen-Erkennung allein aus **Atmung, Herzfrequenz (EKG/PPG/HRV) und Bewegung (Aktigraphie/BCG)** erreicht für 3–4 Stadien (Wake / Light / Deep / REM) ca. **73–86 % Accuracy** (`κ ≈ 0,55–0,70`).
  * **Forderung nach Uncertainty Quantification (Konfidenzschätzung):** Für den praktischen klinischen Einsatz muss die Software anzeigen, bei welchen 30s-Epochen die Klassifikation **unsicher (mehrdeutig)** ist, damit der Arzt im GUI gezielt nur diese Stellen nachprüfen muss.

---

#### Paper 5: Salvi, Loh, Seoni, Barua, García, Molinari & Acharya (2024)
* **Titel:** *Multi-modality approaches for medical support systems: A systematic review of the last decade*
* **Journal & Link:** *Information Fusion*, Vol. 103, 102134, Elsevier (2024) — `https://doi.org/10.1016/j.inffus.2023.102134`
* **Fokus:** Systematisches PRISMA-Review über **81 Q1-Journal-Studien (2012–2022)** zu multimodaler Datenfusion in der Medizin.
* **Kernaussagen & Fusions-Taxonomie:**
  * **3 Datentyp-Kategorien:** (1) *Bioimaging* (MRT, CT, Röntgen), (2) *Biosignals* (EEG, EKG, EOG, EMG, PPG, SpO2), (3) *Mixed* (Kombination aus Biosignalen/Bildern mit **klinischen/demografischen Tabellendaten** wie Alter, Geschlecht, BMI, Fragebögen).
  * **Dokumentierte Forschungslücke im Bereich Schlaf (S. 10):** Die Autoren halten explizit fest: *„Finally, for sleep health, only biosignals-based approaches are utilized, with no consideration for clinical or demographic patient data.“* -> **Wenn ihr in eurem Modell Biosignale (Atmung, SpO2, EKG, EEG) mit klinisch-demografischen Parametern (Alter, Geschlecht, BMI, Schlaflage) kombiniert, schließt ihr exakt diese in Information Fusion (2024) benannte Forschungslücke!**
  * **Zwei Haupt-Fusionsstrategien:**
    1. **Feature-Level Fusion (Early / Intermediate Fusion):** Separate Merkmalsextraktion pro Modalität (z. B. durch modalitätsspezifische 1D-CNN-Encoder), danach Verknüpfung über Concatenation, **Cross-Modal Attention**, **Gated Multimodal Units (GMU)** oder **Graph Neural Networks (GNN)**. Erfasst direkte Wechselwirkungen zwischen den Sensoren.
    2. **Decision-Level Fusion (Late Fusion):** Jede Modalität besitzt einen unabhängigen Klassifikator; die Wahrscheinlichkeits-Outputs werden über Weighted Averaging, Voting oder Stacking fusioniert. Vorteil: Extrem flexibel, wenn einzelne Sensoren fehlen (*Missing Modalities*).

---

#### Paper 6: Secinaro, Calandra, Secinaro, Muthurangu & Biancone (2021)
* **Titel:** *The role of artificial intelligence in healthcare: a structured literature review*
* **Journal & Link:** *BMC Medical Informatics and Decision Making*, Vol. 21, 125 (2021) — `https://doi.org/10.1186/s12911-021-01488-9`
* **Fokus:** Strukturierte Literaturübersicht und bibliometrische Analyse (288 Studien) zu KI im Gesundheitswesen.
* **Kernaussagen für euer Projekt:**
  * Identifiziert 5 Hauptcluster: *Health Services Management, Predictive Medicine, Patient Data, Diagnostics, Clinical Decision-Making*.
  * Diskutiert die Gefahr des **„De-skilling“ von Ärzten** durch blinde Black-Box-Automatisierung sowie die kritische Bedeutung von **Data Quality Management**, **Transparenz / Explainability** und **Datenschutz (Data Privacy / Governance)**. Dies liefert die direkte wissenschaftliche Begründung für eure **lokale, 100 % datenschutzkonforme Offline-Applikation mit visueller XAI-Entscheidungsbegründung**.

---

## 3. Vollständige 18-Spalten-Literatur-Matrix (Direkt im Dokument integriert)

Die folgende Tabelle enthält alle **16 zentralen Referenzwerke** (die 6 Kern-Paper aus eurem Ordner, die 2 Grundlagen-Referenzen der Präsentation sowie 8 hochrelevante SOTA-Paper aus der Web-Recherche) exakt aufbereitet nach den **18 Spalten der Vorlage `Team_project_List_of_useful_tools`**:

| Article# | Read on | Category | Focus | Title | Year | Journal | First author | Main Contents | Input | Main Algorithm | Dataset | Sample size | Methods | Results | Keywords | Comments/Ideas | Review |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 2026-10-19 | Original Research (Core) | OSA Event Detection, AHI & Grad-CAM für Portable Monitor | *Obstructive sleep apnea event detection using explainable deep learning models for a portable monitor* | 2023 | Frontiers in Neuroscience (17:1155900) | Ángel Serrano Alarcón (Co-Autor: Prof. R. Seepold) | Erklärbares 1D-CNN für Portable Monitors (PM) zur OSA-Detektion in 60s-Fenstern, AHI-Berechnung & Grad-CAM-Visualisierung auf 3 NSRR-Datensätzen. | 4 Signale (auf 1 Hz): `SpO2`, `HR`, `Thor-Res` (RIP), `Abdo-Res` (RIP) | 1D-CNN (6 Conv-Layer + BatchNorm + ReLU + Dropout + GAP + Sigmoid); Hyperband-Tuning; Grad-CAM | SHHS2 (Train/Val/Test), SHHS1 (Test), MESA (Test) | SHHS2: 1.631 (1.224 Train, 244 Val, 163 Test); SHHS1: 1.859; MESA: 976 | 60s-Fenster (Label=1 ab >=10s Apnoe); 50/50 Balancierung; Vergleich von 4 Varianten mit/ohne Artefakt-Interpolation (`SpO2<80%`, `HR<40/>200`) und fensterweisem Z-Score; 5-Fold CV. | Model 4 (Artefakt-Filter + Z-Score): SHHS2 Acc=84,3%, Sens=82,5%, Spec=86,0%, AUC=92,1% (`LR+=5,89`); MESA Acc=74,5%, AUC=80,8%; SHHS1 Acc=70,0%, AUC=77,7%. AHI `R²=0,62–0,65`. | obstructive sleep apnea, portable monitor, deep learning, 1D-CNN, Grad-CAM, AHI | **Haupt-Referenz unseres Betreuers!** Genannte Limitationen (S. 17) sind unsere direkte Projekt-Roadmap: (1) Echte TST statt TRT für AHI nutzen, (2) Multiklassen (OSA/CSA/Hypopnoe), (3) Höhere Sampling-Rate (10–100 Hz), (4) Robuste Normalisierung. | Sehr hohe Relevanz (Baseline für WP 2 & WP 5). |
| **2** | 2026-10-19 | Review Article (Core) | AI-Detektion von Schlafapnoe über 5 Signalmodalitäten | *AI-driven approaches for automatic detection of sleep apnea/hypopnea based on human physiological signals: a review* | 2025 | Health Information Science and Systems (13:7) | Dandan Peng | Umfassendes Review über Datenakquise, Denoising, Normalisierung, 1D- vs. 2D-Transformation und Klassifikation für 5 Biosignale. | Nasal Airflow, `SpO2`, EKG (HRV/EDR), EEG (5 Bänder), Snoring Sound | 1D-CNN, 2D-CNN (ResNet, Inception, VGG19, ShuffleNet), BiLSTM, CNN-LSTM, Transformer, SVM, RF | MESA, Apnea-ECG, UCDDB, MIT-BIH, CHAT, PSG-Audio, SHHS1/2 | Review über ~70+ Studien (`N=8` bis `N=8.068`) | Bandpass, Z-Score & Min-Max; 1D-zu-2D-Bilder via CWT-Skalogramm, STFT-Spektrogramm, Gabor-Transform, Recurrence Plots (RP) & Mel-Spektrogramm. | Airflow+`SpO2` (ECA-CNN): Acc=95,91%; `SpO2`+EKG (FNN): Acc=91,83% (vs. 80,04% EKG allein); EKG (CNN-LSTM): Acc=97,21% (Apnea-ECG) & 93,7% (UCDDB); EEG+EKG+Airflow (RP-CNN): Acc=90,45%. | Sleep apnea/hypopnea, Automatic detection, Physiological signals, Multimodal fusion | Liefert exakte Formeln für CWT, STFT, Gabor sowie AASM-Kriterien. Zeigt den hohen Gewinn durch Channel Attention (ECA/SE) bei multimodaler Fusion. | Sehr hohe Relevanz (Methoden- & Preprocessing-Kapitel). |
| **3** | 2026-10-19 | Systematic Review (Core) | EEG-basierte Schlafapnoe-Detektion & Sub-Frame-Analyse | *Sleep Apnea Detection Using EEG: A Systematic Review of Datasets, Methods, Challenges, and Future Directions* | 2025 | Annals of Biomedical Engineering (53:1043–1067) | Shireen Fathima | PRISMA/PICO-Review (2010–2024) über 63 Studien zur Erkennung und Subtypisierung (OSA, CSA, MSA) von Schlafapnoe mittels EEG. | EEG nach 10-20-System (`C3-A2`, `C4-A1`, frontal, okzipital; `100–250 Hz`) | SVM, kNN, Random Forest, Bagged Trees, BiLSTM, CNN-BiLSTM, CNN-GRU, UNETR | UCDDB (25), MIT-BIH (16), Sleep-EDF (197), ISRUC (118), SHHS (6.441), WSC (1.123), CAP (108), CinC 2018 (1.983), PSG-Audio (212) | 63 eingeschlossene Studien (aus 402 Treffern) | 50/60 Hz Notch, 0,5–40 Hz Butterworth IIR; **Sub-Frame-Segmentierung (1–10s)** für Stationarität (ADF-Test); Zerlegung (DWT, VMD, EMD) in Delta, Theta, Alpha, Sigma, Beta, Gamma; Feature-Selektion (ANOVA, mRmR, ReliefF). | Rechts-zentraler Kanal `C4-A1` schlägt `C3-A2` (92,73% vs. 90,05%, BiLSTM) wg. Rechts-Hemisphären-Dominanz; VMD+CNN-BiLSTM: Acc=93,22% (UCDDB) & 93,25% (CinC2018); CNN-GRU (OSA vs. CSA): Acc=99,3%. | Sleep Apnea, Obstructive Sleep Apnea, EEG, Machine Learning, Sub-frame analysis | Begründet die Wahl von `C4-A1`/`C4-M1`, die Nutzung kurzer Sub-Frames (2–10s) für Arousal-Transienten im Beta-/Delta-Band und die Trennung von OSA, CSA und MSA. | Sehr hohe Relevanz (Hauptquelle für WP 3: EEG-Modul). |
| **4** | 2026-10-19 | Review Article (Core) | Automatisches Schlafstadien-Scoring (AASM) | *Current status and prospects of automatic sleep stages scoring: Review* | 2023 | Biomedical Engineering Letters (13:247–272) | Maksym Gaiduk (Co-Autoren: Á. Alarcón, Prof. R. Seepold) | Semi-systematisches Review über 125 Studien (2018–2022) zur automatischen Klassifikation der 5 Schlafstadien (W, N1, N2, N3, REM) in 30s-Epochen. | EEG (69 Studien), EEG+EOG+EMG, EEG+EOG, sowie EKG/HRV, PPG, Atmung, Bewegung/BCG | CNN+BiLSTM, ResNet+BiGRU, SleepTransformer, U-Sleep, LightGBM (YASA), RF+HMM | Sleep-EDFx, MASS, SHHS, ISRUC, MESA, MrOS, CAP, DREAMS, UCDDB, DOD-H/DOD-O | 125 Studien (aus 1.067 Treffern; Mindestgröße Testset >=20 Nächte) | Auswertung nach Signalwahl, 5 vs. 4 vs. 3 Schlafstadien, Subject-independent Validation und Einfluss von Schlafstörungen auf die Genauigkeit. | EEG/EOG/EMG erreicht >90% Acc und Cohen's `κ > 0,80` (menschliches Inter-Rater `κ ≈ 0,76`). Starke Degradation, wenn nur auf Gesunden trainiert und auf Apnoe-Patienten getestet wird! Cardiorespiratory/PPG allein: 73–86% Acc (`κ ≈ 0,55–0,70`). | Automatic sleep assessment, Sleep scoring, Physiological signals, Sleep stages | Zweites Kern-Paper von Prof. Seepold! Fordert Cross-Dataset-Generalisierung, Training auf gemischten Kohorten (Gesunde + Apnoe) und **Uncertainty / Confidence Estimation**. | Sehr hohe Relevanz (Grundlage für Schlafstadien & TST-Berechnung). |
| **5** | 2026-10-19 | Systematic Review (Core) | Multimodale Datenfusion in medizinischen Support-Systemen | *Multi-modality approaches for medical support systems: A systematic review of the last decade* | 2024 | Information Fusion (103:102134, Q1) | Massimo Salvi (zitiert Dr. G. M. Mattia et al., 2022) | PRISMA-Review über 81 Q1-Studien (2012–2022) zu multimodaler Fusion in 9 medizinischen Fachgebieten inkl. Sleep Health. | (1) Bioimaging, (2) Biosignals (EEG, EKG, EOG, EMG, PPG, SpO2), (3) Mixed (Biosignale + klinische/demografische Daten) | Feature-Level Fusion (Concatenation, Cross-Modal Attention, GMU, GNN) vs. Decision-Level Fusion (Voting, Stacking, Weighted Average) | Sleep-EDF (197), CinC 2018 (994), MASS-SS3 (62), ADNI, PPMI, u. a. | 81 Q1-Journal-Studien (aus 1.260 Treffern) | Gegenüberstellung von Early/Intermediate Feature-Fusion und Late Decision-Fusion; Analyse von Harmonisierungs-, Komplexitäts- und XAI-Problemen. | Multimodale Fusion bringt +5% bis +15% Genauigkeitsgewinn gegenüber Einzelmodalitäten (z. B. *SleepPrintNet* 88,8%, *CRPEMA* Arousal AUC 0,844). | Data fusion, Deep learning, Multi-modality, Fusion methods, Healthcare | **Wichtige Lücke (S. 10):** Im Bereich 'Sleep Health' wurden bisher nur reine Biosignale ohne Demografie/Klinikdaten fusioniert -> wir kombinieren Biosignale + Alter/Geschlecht/BMI/Lage ('Mixed Multimodality')! | Sehr hohe Relevanz (Fusions-Taxonomie für WP 4). |
| **6** | 2026-10-19 | Structured Review (Core) | KI im Gesundheitswesen, CDSS, Datenqualität & Ethik | *The role of artificial intelligence in healthcare: a structured literature review* | 2021 | BMC Medical Informatics and Decision Making (21:125) | Silvana Secinaro | Bibliometrische und strukturierte Analyse von 288 Studien zur Rolle von KI bei klinischer Entscheidungsunterstützung, Management und Datenschutz. | Klinische Daten, EHR, Wearable Patient Monitoring Systems (WPMs) | Machine Learning, Deep Learning, Clinical Decision Support Systems (CDSS) | Scopus Bibliometrie-Korpus (1992–2021) | 288 Peer-Review-Artikel (aus 5.069 Treffern) | SLR-Protokoll nach Massaro/PRISMA + R-Paket Bibliometrix (Cluster-Dendrogramm, Keyword-Treemap). | 5 Hauptcluster: Health Services Management, Predictive Medicine, Patient Data, Diagnostics, Clinical Decision-Making. Warnt vor 'De-skilling' von Ärzten durch Black-Box-KI und fordert Datenschutz & Datenqualität. | Artificial intelligence, Healthcare, Patient data, Clinical decision-making | Wissenschaftliche Begründung für unsere **100 % lokale, DSGVO-konforme Offline-App** und das **Human-in-the-Loop XAI-Konzept**, das dem Arzt die Entscheidung vorbereitet statt abnimmt. | Hohe Relevanz für Einleitung, Diskussion & Ethik/DSGVO. |
| **7** | 2026-10-19 | PhD Thesis (Präsentation Folie 17) | Multimodale CNNs & Verständnis des Netzwerkverhaltens (XAI) | *Intelligence artificielle et maladies neurologiques : aider le diagnostic et améliorer la compréhension du comportement des réseaux de neurones convolutifs* | 2022 | Université Toulouse III (2022TOU30282) & Neurocrit. Care | Giulia Maria Mattia (Betreuerin unseres Projekts!) | Multimodale 3D-CNNs zur neurologischen Diagnostik kombiniert mit Methoden zur Analyse und Interpretation des internen CNN-Entscheidungsverhaltens. | Multimodale Neuro-Daten (sMRI, fMRI, klinische Indizes) | Multimodal 3D-CNN + Explainability / Feature Attribution | Klinische Neuro-Kohorte (u. a. Koma-Patienten) | 63 Patienten | Multimodale Fusion kombiniert mit eingehender Untersuchung der gelernten Merkmalsrepräsentationen. | 96% Klassifikationsgenauigkeit durch multimodale Fusion; erhöhte Nachvollziehbarkeit für Kliniker. | Multimodal deep learning, CNN behavior, Explainable AI, Neurology | **Dr. Giulia Maria Mattia ist unsere Betreuerin!** Zeigt, wie wichtig eine saubere Untersuchung des CNN-Verhaltens (Grad-CAM, Filter-/Attention-Analyse) für eine Bestnote ist. | Sehr wichtig für die Ausrichtung von WP 4 und WP 5. |
| **8** | 2026-10-19 | External SOTA Paper | Simultane Schlafstadien- UND Apnoe-Detektion (Multi-Task) | *A Comparison of Signal Combinations for Deep Learning-Based Simultaneous Sleep Staging and Respiratory Event Detection* | 2022 | IEEE Transactions on Biomedical Engineering (69(5):1740–1750) | Riku Huttunen | Multi-Task 1D-U-Net, das gleichzeitig die 5 Schlafstadien UND respiratorische Events (Apnoe, Hypopnoe, Desaturierung) vorhersagt. | 3 Setups: (1) `PPG+SpO2`, (2) `PPG+SpO2+Nasal Pressure`, (3) `SpO2+Nasal Pressure+EEG+Thermistor+RIP` | Multi-Task 1D U-Net (gemeinsamer Encoder, zwei parallele Decoder-Köpfe für Sleep Stages & Apnea Events) | Princess Alexandra Hospital Sleep Disorders Centre | 877 Patienten mit OSA-Verdacht (80/10/10 Split) | Simultanes Sequenz-zu-Sequenz-Training über die gesamte Nacht; Vergleich leichter Home-Sensor-Kombinationen gegen volles PSG-Setup. | Sleep Staging Acc steigt von 69% (Setup 1) und 70% (Setup 2) auf 79% (Setup 3); präzise simultane Apnoe-Erkennung erlaubt echte TST-basierte AHI-Berechnung. | Multi-task learning, Sleep staging, Obstructive sleep apnea, U-Net, Multimodal signals | **Direkte Architektur-Vorlage:** Löst das TRT-vs.-TST-Problem aus Alarcón et al. (2023) durch gemeinsames Vorhersagen von Schlafstadien und Apnoe-Events! | Sehr hohe Relevanz für WP 2, WP 3 und WP 4. |
| **9** | 2026-10-19 | External SOTA Paper | Cross-Dataset-Harmonisierung über 16 Kohorten | *U-Sleep: resilient high-frequency sleep staging* | 2021 | npj Digital Medicine (4:72, Nature Portfolio) | Mathias Perslev | 1D Fully Convolutional U-Net für hochgradig generalisierendes Schlaf-Scoring über 16 verschiedene klinische Kohorten hinweg. | Beliebige Kombination aus 1 EEG + 1 EOG Kanal (einheitlich auf `128 Hz` resampled) | 1D Fully Convolutional Encoder-Decoder Network (U-Time / U-Sleep, 11 Blöcke) | 16 klinische Studien (SHHS, MESA, MrOS, CHAT, ISRUC, UCDDB, Sleep-EDF, CAP, DREAMS u. a.) | 15.660 Probanden (19.924 PSGs) | Resampling auf 128 Hz; **Robuste Per-Subject IQR-Normalisierung** (`(x - Median) / IQR`, Clipping auf `[-20, +20]`); zufällige Kanalwahl im Training. | Zero-Shot Cross-Dataset Kappa `κ ≈ 0,79–0,83` über gesunde und schwer kranke Schlafapnoe-Patienten hinweg. | Sleep staging, Cross-dataset generalization, U-Net, Time-series segmentation, IQR scaling | Wird in Alarcón et al. (2023, S. 17) als zukünftige Lösung für das Windowing-Problem zitiert! Wir übernehmen die **Robuste IQR-Skalierung** direkt für unsere Daten-Harmonisierung (WP 1). | Pflicht-Referenz für WP 1 (Cross-Dataset-Harmonisierung). |
| **10** | 2026-10-19 | External SOTA Paper | Explainable Transformer & Uncertainty Quantification | *SleepTransformer: Automatic Sleep Staging With Interpretability and Uncertainty Quantification* | 2022 | IEEE Transactions on Biomedical Engineering (69(8):2456–2467) | Huy Phan | Hierarchischer Transformer mit Self-Attention-Erklärbarkeit und Unsicherheits-Quantifizierung (Confidence Estimation) für Schlafdaten. | Single-Channel EEG (STFT-Spektrogramm) | Hierarchischer Transformer (Intra-Epoch Self-Attention + Inter-Epoch Sequence Attention) | SHHS1 (5.791) & Sleep-EDF-78 (78 Probanden, 163 PSGs) | 5.869 Probanden | Shannon-Entropie der Softmax-Ausgabe als kalibriertes Unsicherheitsmaß; Attention-Gewichte zur Lokalisierung relevanter Signalabschnitte. | SHHS1: Acc=87,7%, F1=80,1%, `κ=0,83`; Sleep-EDF-78: Acc=81,4%, `κ=0,74`. Hohe Entropie markiert zuverlässig unsichere Übergangs-Epochen für den Arzt. | Transformer, Self-attention, Interpretability, Uncertainty quantification, Sleep staging | Liefert uns die Methode zur **Markierung unsicherer Zeiträume (Confidence Score)** in unserer lokalen GUI (wie in Gaiduk et al., 2023 gefordert). | Hohe Relevanz für WP 5 (XAI & Confidence) und WP 6 (GUI). |
| **11** | 2026-10-19 | External SOTA Paper | Schnelles Open-Source Scoring mit Biosignalen + Demografie | *An open-source, high-performance tool for automated sleep staging (YASA)* | 2021 | eLife (10:e70092) | Raphael Vallat | Extrem schneller LightGBM-Klassifikator auf Zeit-/Frequenz-Features von EEG, EOG, EMG **plus Demografie (Alter, Geschlecht)**. | EEG + EOG + Kinn-EMG (`100 Hz`) + **Alter & Geschlecht** | LightGBM (Gradient Boosted Decision Trees) mit 5-Min.-Kontext-Glättung & SHAP | 8 NSRR- & Dreem-Datensätze (MESA, CFS, CCSHS, SHHS, MrOS, CHAT, HomePAP, DOD-H/O) | 3.798 PSGs (3.163 Train, 585 NSRR-Test, 50 DOD-O OSA-Test) | Hand-crafted Zeit-/Frequenz-/Nichtlineare Features + demografische Metadaten; Auswertung mittels SHAP-Feature-Importance. | 87,5% Acc (`κ=0,82`) auf NSRR-Testset; 84,3% Acc auf OSA-Patienten (DOD-O). Inferenzzeit für 8h Schlaf auf Laptop-CPU: `< 5 Sekunden`! | Open-source, LightGBM, Sleep staging, Demographic metadata, SHAP | Beweist den Nutzen von **Demografie-Merkmalen (Alter, Geschlecht)** und zeigt, wie schnell LightGBM + SHAP lokal auf einer CPU läuft. | Hohe Relevanz für WP 3, WP 4 und WP 6. |
| **12** | 2026-10-19 | External SOTA Paper | Bimodale Feature-Fusion mit ECA Channel Attention | *A bimodal feature fusion convolutional neural network for detecting obstructive sleep apnea/hypopnea from nasal airflow and oximetry signals* | 2024 | Artificial Intelligence in Medicine (150:102808) | Dandan Peng | Dual-Branch 1D-CNN mit Efficient Channel Attention (ECA) zur Fusion von Nasaldruck-Atemfluss und SpO2 für 3-Klassen-Erkennung (Normal / Apnoe / Hypopnoe). | Nasal Pressure Airflow + `SpO2` (10s-Segmente) | Dual-Branch 1D-CNN + Feature Fusion Module mit Efficient Channel Attention (ECA) | FAH & CMH (Klinische PSG-Kohorten) | 450 Patienten (5-Fold CV) | Getrennte 1D-CNN-Feature-Extraktion für Airflow und `SpO2`, danach adaptive Kanalgewichtung über ECA-Attention-Block. | 95,91% Accuracy für 3 Klassen (Normal, Apnoe, Hypopnoe) – deutlich über beiden Einzelmodalitäten. | Bimodal fusion, Channel attention, ECA-Net, Nasal airflow, SpO2, Hypopnea | Konkrete, leichtgewichtige Fusions-Architektur (Dual-/Multi-Branch 1D-CNN + ECA Channel Attention) für unser multimodales Hauptmodell in WP 4. | Hohe Relevanz für WP 4. |
| **13** | 2026-10-19 | External SOTA Paper | Kollaboratives Lernen von Schlafstadien & OSA aus EEG | *EEG-CLNet: Collaborative Learning for Simultaneous Measurement of Sleep Stages and OSA Events Based on Single EEG Signal* | 2023 | IEEE Transactions on Instrumentation and Measurement (72:1–10) | Li Cheng | Multi-Task-Netzwerk (1D-CNN + Transformer + Cross-Task Knowledge Distillation) zur simultanen Erkennung von Schlafstadien und OSA-Events. | Single-Channel EEG | 3x 1D-CNN Blocks + Feature Fusion + Transformer + Cross-Task Knowledge Distillation (CTKD) | UCDDB & Local Hospital Dataset | 60 Patienten | Gemeinsames Feature-Learning mit Cross-Task Knowledge Distillation zwischen Schlafstadien- und OSA-Aufgabe. | 74–75% Acc auf UCDDB allein aus 1-Kanal-EEG (ohne Atmungssensor!); gegenseitige Verbesserung beider Tasks. | Collaborative learning, Multi-task learning, EEG, Sleep stages, OSA detection | Belegt die enge neurophysiologische Kopplung zwischen Schlafstadien und Schlafapnoe. | Gute Zusatzreferenz für WP 3 & WP 4. |
| **14** | 2026-10-19 | External SOTA Paper | Multimodales Sleep Foundation Model & Missing Modalities | *SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals* | 2024 | arXiv / ICML (Stanford University) | Rahul Thapa | Großes multimodales Foundation Model für Schlafdaten, trainiert mit Leave-One-Modality-Out Contrastive Learning über Gehirn-, Herz- und Atmungssignale. | Brain (EEG/EOG/EMG), Cardiac (EKG), Respiratory (Thor/Abdo/Airflow/`SpO2`) | Modality-specific 1D-CNN/ResNet Encoders + Leave-One-Out Contrastive Learning | Stanford Sleep Clinic + NSRR (SHHS u. a.) | >14.000 Patienten (>100.000 Stunden PSG) | Kontrastives Vortraining richtet die Embeddings der 3 Sensorgruppen im selben Latenzraum aus; robust gegen fehlende Kanäle. | Übertrifft unimodale Baselines signifikant bei Schlafstadien- und Apnoe-Klassifikation sowie Transfer Learning. | Foundation model, Contrastive learning, Multimodal sleep analysis, Missing modalities | Motiviert unser **Modality-Dropout-Training** in WP 4, damit unsere App sowohl mit 4-Kanal-PM-Daten als auch mit 12-Kanal-PSG-Daten funktioniert. | Starke SOTA-Referenz für das KES-Paper. |
| **15** | 2026-10-19 | External SOTA Paper | Explainable Ensemble Learning für Heim-Oximetrie | *Ensemble-learning regression to estimate sleep apnea severity using at-home oximetry in adults* | 2021 | Applied Soft Computing (111:107827) | Gonzalo C. Gutiérrez-Tobal (Editor von Alarcón et al., 2023!) | Direkte AHI-Schätzung und Schweregrad-Klassifikation aus Heim-Pulsoximetrie (`SpO2`) mittels Least-Squares Boosting (LSBoost) und erklärbarer Feature-Analyse. | `SpO2` (Zeit-, Frequenz-, ODI- und Entropie-Features) | Least-Squares Boosting (LSBoost) + Explainable Feature Importance | SHHS1 & SHHS2 | >8.000 PSGs (SHHS1 + SHHS2) | Extraktion klinisch interpretierbarer `SpO2`-Merkmale (Desaturierungs-Tiefe, -Dauer, Entropie) zur direkten AHI-Regression. | SHHS1: Acc=89,68%, Sens=87,67%, Spec=79,56%; SHHS2: Acc=88,66%, Sens=94,56%, Spec=64,77%. | Obstructive sleep apnea, Pulse oximetry, Ensemble learning, AHI estimation | Wird in Table 6 von Alarcón et al. (2023) als stärkster Benchmark geführt! Zeigt, wie mächtig explizite `SpO2`-Desaturierungs-Features für die AHI-Schätzung sind. | Wichtiger Benchmark für Table 2 im KES-Paper. |
| **16** | 2026-10-19 | Foundational Book (Folie 17) | Grundlagen KI, Representation Learning & Deep Learning | *Deep Learning* | 2016 | MIT Press (`deeplearningbook.org`) | Ian Goodfellow (Co-Autoren: Y. Bengio, A. Courville) | Standardwerk zu Machine Learning, Representation Learning und Deep Learning (CNNs, RNNs/LSTMs, Autoencoder, Regularisierung). | Allgemeine Datenrepräsentationen | MLP, CNN, RNN/LSTM, Autoencoder, Adam, BatchNorm, Dropout | Theoretisches Grundlagenwerk | N/A | Systematische Abgrenzung: Rule-Based Systems -> Classic ML (Hand-designed features) -> Representation Learning -> Deep Learning. | Liefert die Diagramme auf Folie 9 und 10 der Kick-off-Präsentation (`Team_project_AIN_26WS`). | Artificial intelligence, Machine learning, Representation learning, Deep learning | Pflichtreferenz aus der Kick-off-Präsentation für das Grundlagenkapitel im Paper. | Grundlagen-Referenz (Phase 1). |

---

## 4. Biophysik, Sensorik & Signalgewinnung in der Schlafmedizin (Deep Dive)

Um multimodale Schlafdaten korrekt vorzuverarbeiten, sinnvolle Features zu extrahieren und die XAI-Visualisierungen medizinisch fundiert zu erklären, folgt hier die detaillierte Aufschlüsselung **aller Schlafsensoren**, ihrer **physikalischen Messprinzipien**, ihrer **Abtast- und Signalfrequenzen (in Hz)** sowie ihrer **spezifischen Muster bei Schlafstadien und Schlafapnoe**.

### 4.1 Übersicht der 4 Sensorgruppen einer Polysomnographie (PSG) & Portabler Monitore (PM)

```text
+---------------------------------------------------------------------------------------------------+
|                   MULTIMODALE SCHLAFSENSORIK (PSG & PORTABLE HOME MONITORS)                       |
+---------------------------------------------------------------------------------------------------+
         |                               |                                 |
         v                               v                                 v
+------------------------+   +---------------------------+   +--------------------------------------+
| 1. NEUROPHYSIOLOGIE    |   | 2. KARDIOVASKULÄR         |   | 3. ATMUNG, AKUSTIK & MOTORIK         |
| (Schlafstadien/Arousal)|   | (Autonomes System/Hypoxie)|   | (Mechanische Obstruktion & Lage)     |
+------------------------+   +---------------------------+   +--------------------------------------+
| • EEG (Hirnströme)     |   | • EKG (Herzströme, Lead II)|  | • Thermistor (Mund/Nasen-Temperatur) |
|   10-20 System:        |   |   fs: 200-500 Hz          |   |   Goldstandard für Apnoe (>=90% Drop)|
|   F4-M1, C4-M1, O2-M1  |   |   Band: 0.3-70 Hz         |   |   fs: 25-100 Hz | Band: 0.1-15 Hz    |
|   fs: 100-500 Hz       |   |   Liefert: HR, HRV & EDR  |   |                                      |
|   Band: 0.3-35 (80) Hz |   |                           |   | • Nasaldruck-Kanüle (NPT)            |
|   Amp: 10-100 µV       |   | • Pulsoximetrie (SpO2)    |   |   Goldstandard Hypopnoe (>=30% Drop) |
|                        |   |   Rot (660nm) / IR (940nm)|   |   fs: 25-100 Hz | Band: DC-100 Hz    |
| • EOG (Augendipol)     |   |   fs: 1-25 Hz (Avg <=3s)  |   |                                      |
|   E1-M2, E2-M2         |   |   15-45s Kreislauf-Delay! |   | • RIP-Gurte (Thorax & Abdomen) / EIP |
|   fs: 100-500 Hz       |   |                           |   |   Trennt OSA (paradox) vs. CSA (flach)|
|   Band: 0.3-35 Hz      |   | • Roh-PPG (Pulswelle)     |   |   fs: 10-100 Hz | Band: 0.1-15 Hz    |
|                        |   |   fs: 25-100 Hz           |   |                                      |
| • Submental-EMG (Kinn) |   |   Liefert: PRV & PWA-Drop |   | • Tracheal-/Raum-Mikrofon (8-48 kHz) |
|   fs: 200-500 Hz       |   |   (Vasokonstriktions-     |   | • 3D-Accelerometer / Körperlage      |
|   Band: 10-100 Hz      |   |    Marker für Arousals)   |   |   (Rückenlage vs. Seitenlage, 1-32Hz)|
+------------------------+   +---------------------------+   +--------------------------------------+
```

---

### 4.2 Elektroenzephalographie (EEG) – Hirnaktivität, Frequenzbänder in Hz & Mikro-Events

#### A. Biophysicalisches Messprinzip
Das Oberflächen-EEG misst **extrazelluläre elektrische Feldpotenziale**, die durch die räumliche und zeitliche Summation von **exzitatorischen und inhibitorischen postsynaptischen Potenzialen (EPSPs und IPSPs)** an den apikalen Dendriten tausender parallel zur Kopfoberfläche ausgerichteter **Pyramidenzellen der Großhirnrinde (Kortex, Schicht III und V)** entstehen. Einzelne Aktionspotenziale von Axonen (`~1 ms`) sind zu kurz und nicht synchron genug, um an der Kopfhaut gemessen zu werden; erst wenn Millionen kortikaler Synapsen rhythmisch synchronisiert feuern (gesteuert durch Schrittmacher im **Thalamus** und im **Hirnstamm / Formatio reticularis**), entstehen an der Kopfhaut messbare Wellen im Mikrovolt-Bereich (`10–100 µV`).

#### B. Hardware, Elektrodenplatzierung (10–20-System) & Abtastrate (`fs`)
* **Elektroden:** Nicht-invasive Silber/Silberchlorid-Napfelektroden (`Ag/AgCl`) mit Elektrolytpaste (Übergangsimpedanz `< 5 kOhm`) oder für das Home-Setting **Trocken-Elektroden (Dry Electrodes)**, **Stirn-Klebe-Patches (`Fp1/Fp2/Fpz`)** oder **In-Ear-EEG**.
* **AASM-Standardplatzierung (Internationales 10–20-System):**
  * **Frontal (`F4-M1`, Backup `F3-M2`):** Optimal für langsame **Delta-Wellen (Tiefschlaf N3)** und **K-Komplexe (N2)**.
  * **Zentral (`C4-M1`, Backup `C3-M2`):** Optimal für **Schlafspindeln (N2)**, **Vertex-Wellen (N1)**, **Sägezahnwellen (REM)** und **Apnoe-Detektion** (wobei `C4` über der rechten Hemisphäre nachweislich sensitiver für Schlafapnoe-Veränderungen ist als `C3`!).
  * **Okzipital (`O2-M1`, Backup `O1-M2`):** Optimal über der Sehrinde für den **Alpha-Grundrhythmus (Wachzustand)** und **kortikale Arousals**.
  * *Referenzierung:* Alle Kanäle werden gegen das **gegenüberliegende (kontralaterale) Mastoid (`M1` hinter dem linken Ohr bzw. `M2` hinter dem rechten Ohr; früher Ohrläppchen `A1`/`A2`)** gemessen, um durch den größeren räumlichen Abstand eine höhere Signalamplitude zu erzielen.
* **Abtastrate (`fs`) & Bandbreite:**
  * **AASM-Empfehlung:** `fs = 500 Hz` (Minimum: `200 Hz`; in älteren öffentlichen Datenbanken meist `100 Hz` [Sleep-EDF, WSC], `125 Hz` [SHHS, CHAT] oder `128 Hz` [UCDDB]).
  * **Warum reichen `100–128 Hz` für eure KI-Pipeline völlig aus?** Nach dem **Nyquist-Shannon-Abtasttheorem** (`f_max < fs / 2`) erfasst eine Abtastrate von `100 Hz` alle Frequenzen bis `50 Hz` exakt. Da sämtliche schlafmedizinischen EEG-Wellen zwischen `0,5 Hz` und `40 Hz` liegen, geht bei `100–128 Hz` keinerlei diagnostische Information verloren!
  * **Filterung:** Bandpass `0,3 Hz bis 35 Hz` (bzw. `40–45 Hz`) plus Kerbfilter (**Notch-Filter**) bei `50 Hz` (Europa) bzw. `60 Hz` (USA) gegen Netzbrummen.

#### C. Die 6 Frequenzbänder des EEG im Detail (Hz, Hirnregion, Schlafstadium & Verhalten bei Schlafapnoe)

| Frequenzband | Frequenzbereich (Hz) | Typische Amplitude | Primäre Hirnregion (Topographie) | Physiologische Bedeutung & Zugehöriges Schlafstadium | Spezifische Veränderung bei Schlafapnoe (OSA / CSA) & Arousals |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Delta (`δ`)**<br>*(inkl. Slow Oscillations `< 1 Hz`)* | **0,5 – 4,0 Hz** | Hoch (`> 75 µV`, oft `75–200 µV`) | **Frontal & Zentral** (`F3/F4`, `C3/C4`), thalamokortikale Synchronisation | **Tiefschlaf / Slow-Wave Sleep (Stadium N3):** Wenn `>= 20 %` einer 30s-Epoche (`>= 6 s`) aus Delta-Wellen (`0,5–2 Hz`, Peak-to-Peak-Amplitude `> 75 µV` frontal) bestehen, wird die Epoche als N3 gewertet. Verantwortlich für körperliche Regeneration, Wachstumshormonausschüttung, Immunsystem und glymphatischen Abtransport von Stoffwechselprodukten im Gehirn. Beim wachen Erwachsenen pathologisch (oder Blinzel-/Schwitz-Artefakt!). | • **Über die Gesamtnacht:** OSA-Patienten haben ein massives **Delta-/Tiefschlaf-Defizit**, weil immer wieder Apnoen einsetzen, bevor N3 erreicht wird.<br>• **Innerhalb eines Apnoe-Fensters:** Zu Beginn einer Apnoe fällt die relative `δ`-Band-Power abrupt ab, bleibt während des Atemstillstands niedrig (oder zeigt kurz vor Ende eine hypoxische Verlangsamung) und steigt nach Beendigung der Apnoe wieder an. |
| **2. Theta (`θ`)** | **4,0 – 8,0 Hz** | Mittel (`20–50 µV`) | **Fronto-zentral, temporal & parietal** (`C3/C4`, `F3/F4`, Hippocampus-Kortex-Schleifen) | **Einschlafphase (Stadium N1)**, leichter Schlaf (**N2**) und **REM-Schlaf (Stadium R)**. Im N1 löst Theta den okzipitalen Alpha-Rhythmus ab (oft begleitet von scharfen *Vertex-Wellen* `4–5 Hz` bei `Cz`). Im REM-Schlaf treten charakteristische **Sägezahnwellen (Sawtooth Waves, `2–6 Hz`)** kurz vor schnellen Augenbewegungen auf. Wichtig für synaptische Plastizität und Gedächtnisverarbeitung. | • Erhöhte Theta-Power im Wach-EEG von Apnoe-Patienten ist ein direkter Biomarker für **chronische Tagesschläfrigkeit** und Übermüdung.<br>• Während obstruktiver Apnoen nimmt die fronto-temporale funktionelle Konnektivität im `θ`-Band signifikant ab. |
| **3. Alpha (`α`)** | **8,0 – 12,0 Hz**<br>*(teilw. bis `13 Hz`)* | Mittel (`20–60 µV`) | **Okzipital & Parietal** (`O1/O2`, Sehrinde / Posterior Dominant Rhythm) | **Entspannter Wachzustand bei geschlossenen Augen (Stadium W)**. Sobald eine Person die Augen öffnet oder konzentriert rechnet, wird Alpha blockiert (*Alpha-Blockade* -> Übergang zu Beta). Fällt der Alpha-Anteil in einer 30s-Epoche unter `< 50 %` und weicht gemischten niedrigen Frequenzen (`4–7 Hz`), beginnt definitionsgemäß der Schlaf (**Stadium N1**). | **Der primäre klinische Marker für kortikale Arousals (Weckreaktionen):** Nach AASM ist ein **EEG-Arousal** definiert als ein abrupter Frequenzwechsel hin zu **Alpha (`8–12 Hz`), Theta oder Beta (`> 16 Hz`, keine Spindeln) für mindestens `3 Sekunden`** (nach `>= 10 s` vorausgegangenem Schlaf). Genau dieser Alpha-/Beta-Ausbruch rettet den OSA-Patienten vor dem Ersticken, indem er den Tonus der Rachenmuskulatur wiederherstellt – zerstört aber gleichzeitig die Schlafkontinuität! |
| **4. Sigma (`σ`)**<br>*(Schlafspindel-Band)* | **11,0 – 16,0 Hz**<br>*(klassisch `12–14 Hz`)* | `15–50 µV` (spindelförmig an- und abschwellend) | **Zentral (`C3/C4`)** für schnelle Spindeln (`13–15 Hz`) & **Frontal (`F3/F4`)** für langsame Spindeln (`11–13 Hz`) | **Definierendes Merkmal von Stadium N2 (Leichtschlaf, ca. 45–55 % der Gesamtschlafzeit):** Entsteht durch rhythmische Entladungen im *Nucleus reticularis thalami* (`0,5–3,0 s` Dauer). Schlafspindeln wirken als **„Thalamischer Türsteher“**, der äußere Sinnesreize (Geräusche) vom Kortex abblockt, um den Schlaf zu schützen, und dienen dem Gedächtnistransfer vom Hippocampus in den Neokortex. | • Bei Schlafapnoe sind **Dichte, Frequenz und Amplitude der Schlafspindeln sowie der K-Komplexe im N2-Schlaf signifikant reduziert und fragmentiert**.<br>• Während die *Fuzzy Entropy* in `δ, θ, α` während einer Apnoe sinkt, steigt sie im **Sigma-Band (`12–16 Hz`)** charakteristisch an! |
| **5. Beta (`β`)** | **13,0 – 30,0 Hz**<br>*(außerhalb des Spindelbands oft ab `16 Hz`)* | Niedrig (`< 20 µV`) | **Frontal & Zentral** (`F3/F4`, `C3/C4`, sensomotorischer & präfrontaler Kortex) | **Aktiver, aufmerksamer Wachzustand (Alertness, kognitive Arbeit)** sowie desynchronisierter **REM-Schlaf**. | **Scharfer Energie-Transienten-Peak am Ende einer Apnoe:** Wenn die Atmung aussetzt und das Gehirn den Notfall-Weckreiz auslöst, zeigt das `β`-Band einen **abrupten, steilen Energiesprung (Energy Transient)** (siehe Fathima & Ahmed, 2025, Fig. 9). Das Verhältnis `Beta-Power / Delta-Power` ist daher eines der trennschärfsten EEG-Features zur zeitlichen Lokalisierung von Apnoe-Endpunkten! |
| **6. Gamma (`γ`)** | **30,0 – 80,0 Hz**<br>*(in PSG meist `30–45 Hz` wg. Tiefpass)* | Sehr niedrig (`< 10 µV`) | **Diffus kortikal** (lokale neuronale Ensemble-Synchronisation) | Höhere kognitive Merkmalsbindung, intensive Traumphasen im REM-Schlaf, akute Stress-/Schreckreaktion. | Steigt bei starken **kortikalen und autonomen Arousals** am Ende einer Apnoe zusammen mit Beta stark an. *Technischer Hinweis:* Im Bereich `> 30 Hz` überlagert sich das EEG oft mit **Muskel-Artefakten (EMG der Schläfen-/Kiefermuskulatur)** beim Luftschnappen nach einer Apnoe – für einen Apnoe-Detektor ist genau dieser kombinierte EEG-/EMG-Burst jedoch ein hochgradig nützliches Signal! |

#### D. Die 5 Schlafstadien nach AASM (v2.6) im Überblick (30-Sekunden-Epochen)
1. **Stadium W (Wake / Wach):** `> 50 %` der 30s-Epoche zeigen okzipitalen Alpha-Rhythmus (`8–12 Hz`, bei geschlossenen Augen) oder schnelle Beta-Aktivität mit Lidschlägen (`0,5–2 Hz`), schnellen Augenbewegungen und hohem Kinn-EMG-Tonus.
2. **Stadium N1 (NREM 1 / Einschlafstadium, ~5 % der Nacht):** Alpha-Rhythmus verschwindet (`< 50 %`), ersetzt durch **niedrigamplitudige gemischte Frequenzen (LAMF, `4–7 Hz` Theta)**, **langsame pendelnde Augenbewegungen (SEMs)** im EOG und **Vertex Sharp Waves** bei `Cz`.
3. **Stadium N2 (NREM 2 / Leichtschlaf, ~45–55 % der Nacht):** Auftreten von mindestens einem **K-Komplex** (scharfe negativ-positive Welle `>= 0,5 s`, frontal maximal) und/oder einer **Schlafspindel** (`11–16 Hz` Burst `>= 0,5 s`, zentral maximal) bei weniger als `< 20 %` Delta-Wellen.
4. **Stadium N3 (NREM 3 / Tiefschlaf / Slow-Wave Sleep, ~15–25 % der Nacht):** Mindestens **`>= 20 %` der 30s-Epoche (`>= 6 Sekunden`)** bestehen aus langsamen, hochamplitudigen **Delta-Wellen (`0,5–2,0 Hz`, Amplitude `> 75 µV` frontal)**. Augenbewegungen fehlen, Kinn-EMG ist niedrig. *(Hinweis: Da im N3-Schlaf der zentrale Atemantrieb sehr stabil ist und die obere Atemwegsmuskulatur ruhiger arbeitet, treten im N3-Schlaf deutlich weniger Apnoen auf als im N1-, N2- und REM-Schlaf!)*
5. **Stadium R (REM-Schlaf / Traumschlaf, ~20–25 % der Nacht):** Gekennzeichnet durch die klassische Trias:
   * **Desynchronisiertes EEG** (niedrigamplitudige gemischte Theta-/Beta-/Alpha-Frequenzen + *Sawtooth Waves* `2–6 Hz`),
   * **Schnelle, ruckartige Augenbewegungen (REMs)** im EOG,
   * **Komplette Muskelatonie (minimaler Tonus)** im submentalen Kinn-EMG.
   * **Klinische Bedeutung für Schlafapnoe:** Im REM-Schlaf ist die Muskelerschlaffung der Zungen- und Rachenmuskulatur am stärksten und die Empfindlichkeit des Atemzentrums auf CO2 reduziert -> **im REM-Schlaf dauern obstruktive Apnoen am längsten und führen zu den gefährlichsten Sauerstoffabfällen (`SpO2 < 80 %`)!**

---

### 4.3 Elektrookulographie (EOG) & Elektromyographie (EMG)
* **EOG (Augenbewegungen):**
  * **Prinzip:** Misst das permanente **korneo-retinale Bestandspotenzial** des Auges (Hornhaut vorne positiv `+0,4 bis +1,0 mV` gegenüber Netzhaut hinten negativ).
  * **Setup & Frequenz:** 2 Elektroden (`E1-M2` links unten außen, `E2-M2` rechts oben außen), `fs = 100–500 Hz`, Bandpass `0,3–35 Hz`, Amplitude `50–350 µV`.
  * **Nutzen:** Gegenphasige Ausschläge zwischen `E1` und `E2` beweisen echte Augenbewegungen (Slow Eye Movements `0,2–0,5 Hz` in N1 vs. schnelle REMs in Stadium R), während gleichphasige Ausschläge eingestreute Stirn-Delta-Wellen (N3) anzeigen.
* **Submentales EMG (Kinntonus) & Tibialis-EMG (Beine):**
  * **Setup & Frequenz:** 3 Elektroden am Kinn bzw. 2 Elektroden am Schienbein (*M. tibialis anterior*), `fs = 200–500 Hz`, Bandpass **`10–100 Hz`** (Hochfrequenz-Muskelpotenziale).
  * **Nutzen:** Essenziell zur sicheren Erkennung der **REM-Atonie**, zur Erkennung von **REM-Arousals** (erfordern laut AASM zwingend einen Kinn-EMG-Anstieg `>= 1 s`), zur Erfassung des muskulären Luftschnappens am Ende einer obstruktiven Apnoe sowie (über das Bein-EMG) zur Diagnose von **Periodic Limb Movements in Sleep (PLMS)**.

---

### 4.4 Elektrokardiographie (EKG), Herzratenvariabilität (HRV) & ECG-Derived Respiration (EDR)
* **Messprinzip & Frequenz:** Aufzeichnung der elektrischen Herzaktivität (P-Welle, QRS-Komplex, T-Welle) über eine Einkanal-Brustwandableitung (modifizierte Ableitung II), `fs = 100–500 Hz`, Bandpass `0,3–70 Hz`.
* **Die 3 Informationskanäle des EKG für die Schlafapnoe-Detektion:**
  1. **CVHR (Cyclical Variation of Heart Rate – Bradykardie/Tachykardie-Schaukel):**
     * Während des Atemstillstands (Apnoe) dominiert der **parasympathische Tauchreflex (Vagusnerv)** -> die Herzfrequenz fällt progressiv ab (**Bradykardie**, z. B. auf `45–55 bpm`, die R-R-Abstände werden länger).
     * Beim rettenden Arousal und ersten tiefen Atemzug feuert der **Sympathikus** -> die Herzfrequenz springt schlagartig nach oben (**Tachykardie**, z. B. auf `75–95 bpm`, die R-R-Abstände werden kurz).
  2. **HRV-Spektralbänder (aus der Zeitreihe der R–R-Intervalle):**
     * **Zeitbereich:** `MeanNN`, `SDNN`, `RMSSD`, `NN50`, `pNN50`.
     * **VLF-Band (Very Low Frequency, `0,0033 – 0,04 Hz`, Periodendauer `25–300 s`):** **Das wichtigste EKG-Frequenzband für Schlafapnoe!** Da sich Apnoe-Episoden alle `30 bis 90 Sekunden` wiederholen, erzeugt die zyklische Bradykardie-Tachykardie-Schwankung einen **markanten Leistungs-Peak genau im VLF-Band (`~0,01–0,03 Hz`)**!
     * **LF-Band (Low Frequency, `0,04 – 0,15 Hz`):** Sympathikus- und Baroreflex-Aktivität (bei OSA durch den nächtlichen Stress stark erhöht).
     * **HF-Band (High Frequency, `0,15 – 0,40 Hz`, entspricht der normalen Atemfrequenz von `9–24 Atemzügen/Min.`):** Normale respiratorische Sinusarrhythmie (Parasympathikus). Das **LF/HF-Verhältnis** ist bei Apnoe-Patienten deutlich erhöht.
  3. **EDR (ECG-Derived Respiration – Atmung direkt aus dem EKG):**
     * Durch die Ausdehnung des Brustkorbs und die Kippung der elektrischen Herzachse beim Ein- und Ausatmen schwankt die **Amplitude der R-Zacken** im Rhythmus der Atmung. Interpoliert man die Spitzenwerte der R-Zacken über die Zeit, erhält man eine abgeleitete Atmungskurve (**EDR**) allein aus dem EKG-Kabel!

---

### 4.5 Pulsoximetrie (`SpO2`) & Photoplethysmographie (`PPG`)
* **Physikalisches Prinzip:** Optische Durchleuchtung/Reflexion am Finger, Ohrläppchen oder Handgelenk mit **rotem Licht (`660 nm`, stark absorbiert von sauerstoffarmem Hb)** und **infrarotem Licht (`940 nm`, stark absorbiert von sauerstoffreichem HbO2)**. Aus dem Quotienten der Wechsel-/Gleichstromanteile beider Wellenlängen wird die arterielle Sauerstoffsättigung `SpO2` in `%` bestimmt.
* **Abtastrate & Mittelungszeit:** `SpO2` wird meist mit `1 Hz` bis `8–25 Hz` gespeichert (wichtig: interne Gerätemittelung `<= 3 Sekunden` bei `60 bpm`, um kurze Entsättigungen nicht zu glätten); die rohe **PPG-Pulswelle** wird mit `25–100 Hz` abgetastet.
* **Klinische Kriterien & die Kreislauf-Totzeit (Circulatory Delay):**
  * Ein Abfall der `SpO2`-Kurve um **`>= 3 %`** (empfohlene AASM-Regel) bzw. **`>= 4 %`** (alternative AASM-Regel) gegenüber der vorangehenden Baseline definiert den **Oxygen Desaturation Index (ODI3% / ODI4%)** und ist das Hauptkriterium zur Bestätigung einer **Hypopnoe**.
  * **Wichtiger physikalischer Effekt für euer ML-Modell:** Da das Blut von der Lunge bis in die Fingerspitze ca. **`15 bis 45 Sekunden` Laufzeit** benötigt, erscheint das `SpO2`-Tal im Signal **immer zeitverzögert nach dem eigentlichen Atemstillstand**!
  * **PPG-Pulswellenamplitude (PWA-Drops):** Beim Arousal am Ende einer Apnoe verengen sich durch den Sympathikus-Reiz die Fingergefäße (Vasokonstriktion) -> die Amplitude der rohen PPG-Welle bricht kurzzeitig stark ein (**PWA-Drop**), was als hervorragender **Ersatz-Marker für EEG-Arousals** dient, wenn kein EEG am Kopf getragen wird!

---

### 4.6 Respiratorische Sensoren – Atemfluss (Thermistor vs. Nasaldruck) & Atemanstrengung (RIP / EIP)

| Sensortyp | Messort & Physikalisches Prinzip | Abtastrate (`fs`) & Filter | Primäre klinische Aufgabe nach AASM |
| :--- | :--- | :--- | :--- |
| **1. Oronasaler Thermistor / Thermocouple** | Sitzt vor Nase UND Mund. Misst die **Temperaturdifferenz** zwischen kühler Einatemluft (`~20°C`) und warmer Ausatemluft (`~37°C`). | `fs = 25–100 Hz`<br>Filter: `0,1–15 Hz` | **Goldstandard zur Detektion von APNOEN:** Da er Nase und Mund gleichzeitig überwacht, erkennt er zuverlässig einen **kompletten Atemfluss-Stopp (`>= 90 %` Amplitudenabfall für `>= 10 s`)**. Wegen thermischer Trägheit nichtlinear bei kleinen Schwankungen. |
| **2. Nasaldruck-Transducer (NPT / Nasal Cannula)** | Nasenbrille misst den **aerodynamischen Staudruck** (`ΔP ~ Flow²`) in den Nasenlöchern über einen Differenzdruckwandler. | `fs = 25–100 Hz`<br>Filter: `DC/0,03–100 Hz` | **Goldstandard zur Detektion von HYPOPNOEN (`>= 30 %` Abfall für `>= 10 s` + `>= 3 %` SpO2-Drop oder Arousal)** sowie **Inspiratory Flow Limitation (IFL / Abflachung der Einatemkurve)** und **RERA**. |
| **3. RIP-Gurte (`Thor-Res` & `Abdo-Res`)** | Zwei elastische Gurte um **Brustkorb (Thorax)** und **Bauch (Abdomen)** mit sinusförmig eingewebter Drahtspule; misst die **Änderung der magnetischen Selbstinduktivität `ΔL`** proportional zur Querschnittsfläche. | `fs = 10–100 Hz`<br>Filter: `0,1–15 Hz` | **Goldstandard zur Unterscheidung von OSA vs. CSA vs. MSA:**<br>• **OSA (Obstruktiv):** Airflow fällt auf `<= 10 %`, aber `Thor-Res` und `Abdo-Res` arbeiten weiter (oft **gegenphasig / paradoxe Schaukelatmung** gegen den verschlossenen Rachen!).<br>• **CSA (Zentral):** Airflow fällt auf `<= 10 %` UND **`Thor-Res` sowie `Abdo-Res` sind komplett flach** (kein Atemantrieb vom Gehirn).<br>• **MSA (Gemischt):** Erst flache CSA-Phase, dann einsetzende OSA-Anstrengung. |
| **4. Elektrische Impedanzpneumographie (EIP)** | Misst über normale EKG-Elektroden mittels hochfrequentem Schwachstrom (`20–100 kHz`) die **elektrische Impedanzänderung des Brustkorbs** durch das Luftvolumen in der Lunge. | `fs = 10–100 Hz` | Komfortabler Ersatz für RIP-Gurte in **Portablen Monitoren (PMs)** (wird u. a. im Ziel-PM des HTWG UC Lab verwendet, siehe Alarcón et al., 2023!). |

---

### 4.7 Akustik (Schnarch-/Trachealmikrofon) & Körperlage / Aktigraphie
* **Tracheal- und Raummikrofon (`fs = 8.000 – 48.000 Hz`):**
  * Erfasst Gewebevibrationen der verengten oberen Atemwege (**Schnarchen**, Hauptenergie zwischen `100 Hz` und `1.000 Hz`, wichtige Features: **MFCCs**, **800-Hz-Power-Ratio**, **Spectral Entropy**).
  * Charakteristisches OSA-Muster: Lautes rhythmisches Schnarchen -> **absolute akustische Stille während der Apnoe (`10–60 s`)** -> **explosives Post-Apnoe-Schnarchen / Luftschnappen (PAS)** beim Arousal.
* **Körperlagesensor & 3-Achsen-Accelerometer (`fs = 1 – 100 Hz`):**
  * Bestimmt über den Erdgravitationsvektor die Schlaflage (**Rückenlage [Supine]**, Seitenlage links/rechts [Lateral], Bauchlage [Prone]).
  * Da bei über 50 % der Patienten der Zungengrund vor allem in **Rückenlage** nach hinten fällt (**Positional OSA**), liefert die Korrelation von erkannten Apnoe-Events mit der Körperlage einen enormen diagnostischen und therapeutischen Mehrwert für eure Software!

---

## 5. Öffentliche Datensätze & 6-Stufen-Pipeline zur Cross-Dataset-Harmonisierung

### 5.1 Übersicht der besten öffentlichen Datensätze für euer Projekt

#### Gruppe A: Sofort frei herunterladbar (Ohne Antrag / Wartezeit – ideal für den sofortigen Start!)
1. **UCDDB – St. Vincent's University Hospital / University College Dublin Sleep Apnea Database (PhysioNet):**
   * **Link:** `https://physionet.org/content/ucddb/1.0.0/`
   * **Umfang:** `25` vollständige Übernacht-PSGs (`6–8 h`) von Erwachsenen mit Verdacht auf Schlafapnoe (`AHI 1,7 bis 90`).
   * **Signale (`128 Hz`):** Volle PSG (14 Kanäle): EEG (`C3-A2`, `C4-A1`), EOG (2 ch), Kinn-EMG, EKG, Oronasaler Thermistor-Airflow, `Thor-Res` (RIP), `Abdo-Res` (RIP), `SpO2`, Pulsrate, Schnarchsensor, Körperposition.
   * **Annotationen:** Sekundengenaue Startzeit & Dauer für **OSA, CSA, MSA, Hypopnoe** sowie **30s-Schlafstadien** und **Demografie (Alter, BMI, Geschlecht, AHI)**!
2. **PhysioNet / Computing in Cardiology Challenge 2018 („You Snooze You Win“ / Massachusetts General Hospital):**
   * **Link:** `https://physionet.org/content/challenge-2018/1.0.0/`
   * **Umfang:** **`1.983` Patienten** (`994` vollständig annotierte Trainings-PSGs + `989` Test-PSGs; einzelne Patientenordner können gezielt per Python-Skript heruntergeladen werden, man muss nicht die gesamten 140 GB auf einmal laden!).
   * **Signale (alle einheitlich bei `200 Hz`!):** 6 EEG-Kanäle (`F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1`), EOG (`E1-M2`), Kinn-EMG, Abdomen, Chest (`Thor`), Airflow, `SaO2`, EKG.
   * **Annotationen:** **AASM-Schlafstadien (W, N1, N2, N3, R)** + exakte Event-Grenzen für **OSA, CSA, MSA, Hypopnoe** UND **RERA / Non-Apnea Arousals**!
3. **ISRUC-Sleep (University of Coimbra, Portugal):**
   * **Link:** `https://sleeptight.isr.uc.pt/`
   * **Umfang:** `118` Probanden (*Subgroup 1:* `100` Patienten mit Schlafstörungen/Apnoe; *Subgroup 2:* `8` Patienten mit je 2 Nächten; *Subgroup 3:* `10` Gesunde).
   * **Signale (`200 Hz`):** 6 EEG, 2 EOG, Kinn-EMG, Bein-EMG, EKG, Airflow, `Thor-Res`, `Abdo-Res`, `SpO2`, Schnarchmikrofon, Körperlage + klinische Patienten-Metadaten.
   * **Annotationen:** AASM-Schlafstadien von **2 unabhängigen Experten** + Apnoe-/Arousal-Events.
4. **PSG-Audio (Sismanoglio General Hospital Athen, ScienceDB):**
   * **Link:** `https://www.scidb.cn/en/detail?dataSetId=778740145531650048`
   * **Umfang:** `212` komplette PSGs (`200 Hz`) von Apnoe-Patienten inkl. synchronisiertem `48 kHz` Tracheal- und Raummikrofon sowie exakten Annotationen für **OSA, CSA, MSA, Hypopnoe, RERA** und Schlafstadien.
5. **PhysioNet Apnea-ECG (`70` Aufnahmen, `8` davon mit vollen Atmungssignalen + `SpO2` bei `100 Hz`):** `https://physionet.org/content/apnea-ecg/1.0.0/`
6. **MIT-BIH Polysomnographic Database (`slpdb`, `16` Aufnahmen bei `250 Hz` mit Schlafstadien & Apnoe-Subtypen):** `https://physionet.org/content/slpdb/1.0.0/`
7. **Sleep-EDF Expanded (`197` PSGs bei `100 Hz`, Standard-Benchmark für Schlafstadien):** `https://physionet.org/content/sleep-edfx/1.0.0/`

#### Gruppe B: Große klinische Kohorten im NSRR (`sleepdata.org` – kostenloser Account + DUA-Antrag)
* **SHHS1 (`5.793` PSGs) & SHHS2 (`2.651` PSGs):** `https://sleepdata.org/datasets/shhs`
* **MESA (`2.060` PSGs):** `https://sleepdata.org/datasets/mesa`
* **CHAT (`1.638` pädiatrische PSGs)** & **MrOS (`2.907` PSGs)**
*(Praxis-Tipp: Da Prof. Seepold und Á. Serrano Alarcón in ihrer Arbeit 2023 exakt SHHS1, SHHS2 und MESA verwendet haben, fragt beim ersten Treffen direkt nach, ob das UC Lab euch die bereits heruntergeladenen SHHS-/MESA-Daten bereitstellen kann, und startet parallel sofort mit UCDDB + CinC 2018 + ISRUC!)*

---

### 5.2 Die 6-Stufen-Pipeline zur Harmonisierung und Kombination mehrerer Datensätze

Wenn ihr mehrere Datensätze (z. B. **UCDDB + CinC 2018 + ISRUC + SHHS2 + MESA**) zu einem großen Trainingsdatensatz verbindet, unterscheiden sich Kanalnamen, Netzfrequenzen (`50 Hz` EU vs. `60 Hz` USA), Abtastraten (`1 Hz` bis `250 Hz`), Verstärkungsfaktoren und Annotationsformate.

Damit das Modell **echte physiologische Muster** statt datensatzspezifischer Artefakte (*Shortcut Learning*) lernt, wendet ihr diese **6-stufige Harmonisierungs-Pipeline** an:

```text
[Roh-Datensätze: UCDDB (128Hz, EU), CinC2018 (200Hz, US), ISRUC (200Hz, EU), SHHS/MESA (125/1Hz, US)]
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  | STUFE 1: Kanonisches Kanal-Mapping & Einheiten-Standardisierung                   |
  | • Mapping auf feste Namen: EEG_C4, EEG_C3, EOG, EMG, ECG, FLOW, THOR, ABDO, SPO2  |
  | • Umrechnung aller Spannungen auf µV (EEG/EOG/EMG) bzw. mV (ECG) und % (SpO2)     |
  +-----------------------------------------------------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  | STUFE 2: Spektrale Bereinigung (Eliminiert den 50-Hz vs. 60-Hz Länder-Bias!)      |
  | • Zero-Phase Butterworth Bandpass (4. Ordnung, vorwärts-rückwärts sosfiltfilt):   |
  |   - EEG / EOG / ECG: 0,3 Hz bis 40,0 Hz (entfernt SOWOHL 50 Hz ALS AUCH 60 Hz!)   |
  |   - Atmung (FLOW, THOR, ABDO): 0,05 Hz bis 3,0 Hz                                 |
  +-----------------------------------------------------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  | STUFE 3: Multi-Rate Polyphasen-Resampling (mit Anti-Aliasing FIR-Filter)          |
  | • Elektrophysiologie (EEG, EOG, EMG, ECG) -> einheitlich auf 100 Hz (oder 128 Hz) |
  | • Atmung (FLOW, THOR, ABDO)               -> einheitlich auf 10 Hz                |
  | • Pulsoximetrie & Herzfrequenz (SPO2, HR) -> einheitlich auf 1 Hz                 |
  +-----------------------------------------------------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  | STUFE 4: Artefakt-Interpolation & Physiologisch korrekte Amplituden-Normierung    |
  | • Artefakt-Clipping & lineare Interpolation: SpO2 outside [65%, 100%], HR [35,220]|
  | • Für EEG/EOG/EMG/ECG/THOR/ABDO/FLOW: Robuste Per-Subject IQR-Skalierung (U-Sleep)|
  |   x_norm = clip( (x - Median_Nacht) / (P75_Nacht - P25_Nacht), -20, +20 )         |
  | • Für SpO2 (NICHT blind fensterweise Z-scoren!):                                  |
  |   Kanal A: (SpO2 - 90.0) / 10.0  |  Kanal B: Delta_SpO2 = SpO2(t) - P95(t-120..t) |
  +-----------------------------------------------------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  | STUFE 5: Vereinheitlichte 1-Hz-Label-Vektoren & AASM-Regel-Harmonisierung         |
  | • Schlafstadien (30s): R&K S3+S4 -> AASM N3; {0:W, 1:N1, 2:N2, 3:N3, 4:REM}       |
  | • Atmungs-Events (1 Hz Maske): {0:Normal, 1:OSA, 2:CSA, 3:MSA, 4:Hypopnoe}        |
  +-----------------------------------------------------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  | STUFE 6: Training mit Modality Dropout & Strengem Subject-/Dataset-Split          |
  | • Zufälliges Ausblenden einzelner Kanäle (p=0,25) + Kanal-Präsenz-Maske m ∈ {0,1} |
  | • GroupKFold nach patient_id + Leave-One-Dataset-Out (LODO) Generalisierungstest  |
  +-----------------------------------------------------------------------------------+
```

#### Warum Stufe 4 (die getrennte Behandlung von `SpO2` vs. Elektrophysiologie/RIP) für euer Paper so wichtig ist:
* In **Alarcón et al. (2023)** wurde jedes 60s-Fenster aller Kanäle einzeln mit `z = (X - μ_fenster) / σ_fenster` normiert.
* **Das physikalische Problem bei `SpO2`:** Wenn ein völlig gesunder Schläfer in einem 60s-Fenster konstant zwischen `97,0 %` und `97,2 %` liegt (`σ_fenster = 0,08 %`), teilt der fensterweise Z-Score durch `0,08 %` und vergrößert dieses bedeutungslose `0,2 %`-Rauschen auf denselben Zahlenbereich (`-2 bis +2`) wie einen echten `6 %`-Apnoe-Abfall von `96 %` auf `90 %`! Genau das führte in Alarcón et al. (2023, Fig. 10) zu Fehlalarmen bei kleinen Schwankungen (`< 3 %`) und zur Überschätzung des AHI bei Gesunden.
* **Eure Verbesserung:** Indem ihr für `SpO2` die **lokale Abweichung zur 120-Sekunden-Baseline (`Delta_SpO2(t) = SpO2(t) - P95(SpO2[t-120 : t])`)** übergebt und für alle anderen Kanäle (`EEG`, `ECG`, `Thor-Res`, `Abdo-Res`, `Airflow`) die ausreißer-robuste **Per-Subject IQR-Skalierung (`x_norm = (x - Median) / IQR`)** nach *U-Sleep (Perslev et al., 2021)* nutzt, bleibt die klinische `3 %`-Desaturierungsschwelle exakt erhalten UND Verstärkerunterschiede zwischen verschiedenen Krankenhäusern werden perfekt herausgerechnet!

---

## 6. Konzept & Architektur eurer lokalen, DSGVO-konformen Ziel-Software

### 6.1 End-to-End Systemarchitektur (100 % Lokal, Offline & Benutzerfreundlich)

```text
+---------------------------------------------------------------------------------------------------+
|                     LOKALE DESKTOP-/LOCALHOST-GUI (PySide6 oder Streamlit)                        |
|                    100 % Offline • Keine Cloud • DSGVO-/Privacy-by-Design                         |
+---------------------------------------------------------------------------------------------------+
  | 1. Datei-Upload (.edf / .edf+ / .csv)                                                         ^
  v                                                                                               |
+-----------------------------------------------------------------------------------------------+ |
| SCHRITT 1: AUTOMATISCHER PREPROCESSOR & SIGNAL QUALITY CHECKER                                | |
| • Erkennt vorhandene Kanäle automatisch (z.B. 4-Kanal-Home-PM oder 12-Kanal-Klinik-PSG)       | |
| • Interpoliert Sensor-Dropouts, berechnet HR & EDR falls nur Roh-EKG vorliegt                 | |
| • Führt 0.3-40 Hz Filterung, Robuste IQR-Skalierung und Delta-SpO2-Berechnung durch           | |
+-----------------------------------------------------------------------------------------------+ |
  |                                                                                               |
  v                                                                                               |
+-----------------------------------------------------------------------------------------------+ |
| SCHRITT 2: DUAL-TASK MULTIMODAL INFERENCE ENGINE (PyTorch / ONNX Runtime auf lokaler CPU/GPU) | |
|                                                                                               | |
|   +-----------------------------------------+   +-----------------------------------------+   | |
|   | MODUL A: Respiratorischer Event-Detektor|   | MODUL B: Schlafstadien- & TST-Schätzer  |   | |
|   | • Multi-Branch 1D-CNN/ResNet +          |   | • Erkennt pro 30s-Epoche:               |   | |
|   |   ECA Channel Attention + BiLSTM/Transf.|   |   Wake (W), N1, N2, N3, REM             |   | |
|   | • Klassifiziert pro Fenster/Sekunde:    |   | • Berechnet die EXAKTE Schlafzeit:      |   | |
|   |   Normal vs. OSA vs. CSA vs. Hypopnoe   |   |   TST = Summe aller Nicht-Wach-Epochen  |   | |
|   +-----------------------------------------+   +-----------------------------------------+   | |
|                        \                                           /                          | |
|                         v                                         v                           | |
|   +---------------------------------------------------------------------------------------+   | |
|   | MODUL C: KLINISCHE DIAGNOSE-SYNTHESE & EXPLAINABLE AI (XAI) ENGINE                    |   | |
|   | • Präziser AHI = (Anzahl Apnoen + Hypopnoen im Schlaf) / Echte Schlafzeit (TST in h)  |   | |
|   | • Schweregrad: Normal (<5), Mild (5-15), Moderat (15-30), Schwer (>=30 Events/h)      |   | |
|   | • 1D-Grad-CAM: Berechnet für jedes Event die zeitliche Heatmap L_GradCAM(t) (Sek. 0-60)|  | |
|   | • SHAP / Channel Attention: Berechnet pro Event den %-Anteil jedes Sensors            |   | |
|   | • Confidence Score (1 - normierte Entropie): Markiert unsichere Epochen für den Arzt  |   | |
|   +---------------------------------------------------------------------------------------+   | |
+-----------------------------------------------------------------------------------------------+ |
  |                                                                                               |
  +---> 2. Übergabe von Diagnose, Event-Zeitstempeln, Grad-CAM-Heatmaps & Klartext-Erklärung -----+
```

### 6.2 Wie die Benutzeroberfläche (GUI) deine Anforderungen umsetzt
1. **Einfaches Laden & Automatische Kanalerkennung:**
   Der Nutzer zieht eine `.edf`- oder `.csv`-Datei per Drag-and-Drop in das Fenster. Das Programm zeigt sofort an, welche Sensoren erkannt wurden (z. B. `SpO2`, `HR`, `Thor-Res`, `Abdo-Res`, `EEG C4-A1`) und wie gut die Signalqualität in Prozent ist.
2. **Klare Gesamtdiagnose auf einen Blick („Ob eine Schlafkrankheit vorliegt“):**
   * **Diagnose-Karte:** z. B. **„Verdacht auf Moderate Obstruktive Schlafapnoe (AHI: 21,4 Events/h)“** (Farb-Ampel: Grün `< 5`, Gelb `5–15`, Orange `15–30`, Rot `>= 30`).
   * **Direkter Vergleich `AHI (über echte Schlafzeit TST)` vs. `REI (über Aufnahmezeit TRT)`:** Zeigt transparent, warum die Schlafzeit-Korrektur (TST = `6,5 h` statt TRT = `8,0 h`) die Diagnose präziser macht.
   * **Subtypen-Aufschlüsselung:** Wie viele Events waren *Obstruktive Apnoen (OSA)*, *Zentrale Apnoen (CSA)* oder *Hypopnoen*, und wie verteilen sie sich auf **REM- vs. NREM-Schlaf** sowie **Rücken- vs. Seitenlage**?
3. **Zeitraum-Lokalisierung („In welchem Zeitraum der Sensordaten das erkenntlich ist“):**
   * **Interaktiver 8-Stunden-Zeitstrahl:** Zeigt das **Hypnogramm** (Schlafphasen über die Nacht) und direkt darunter eine **farbige Event-Leiste** mit allen erkannten Apnoe-/Hypopnoe-Zeiträumen (z. B. `01:42:00–01:43:00`, `02:14:00–02:15:00` usw.) sowie einer **Confidence-Spur** (wo das Modell unsicher war).
   * **Klickbare Event-Tabelle:** Ein Klick auf ein Event in der Liste springt im Signal-Viewer sofort auf die exakte Minute und Sekunde der Aufnahme.
4. **Visuelle & Textuelle Erklärung („Wieso das so ist“):**
   * **1D-Grad-CAM-Overlay direkt hinter den Signalkurven:** Im 60-Sekunden-Detailfenster hinterlegt eine rot-gelbe Heatmap exakt die Sekunden auf den Kurven (`SpO2`, `HR`, `Thor-Res`, `Abdo-Res`, `Airflow`, `EEG`), die den Ausschlag gegeben haben.
   * **Sensor-Gewichtung (Channel Contribution Bar) & Automatischer Befundtext:**
     Für jedes angeklickte Zeitfenster generiert das Programm automatisch eine verständliche Begründung, z. B.:
     > *„Zeitraum **02:14:10 – 02:14:42 Uhr** klassifiziert als **Obstruktive Apnoe (Sicherheit: 94 %)**:*
     > *1. **Atemfluss & Anstrengung (38 % Gewicht):** Abfall des Atemflusses um >90 % über 32 s bei gleichzeitig weiterlaufender, gegenphasiger Bewegung von Brust- und Bauchgurt (typisches obstruktives Muster).*
     > *2. **Sauerstoffsättigung (34 % Gewicht):** Zeitverzögerter Abfall des SpO2-Wertes um 4,6 % (von 96,0 % auf 91,4 % ab Sekunde 28).*
     > *3. **Herzfrequenz & EEG-Arousal (28 % Gewicht):** Bradykardie (52 bpm) während des Atemstillstands mit anschließendem Tachykardie-Sprung auf 79 bpm und hochfrequentem Beta-Arousal im EEG bei Sekunde 42.“*

---

## 7. Arbeitsverteilung für 6 Personen & Wochenzeitplan bis Ende Dezember 2026

### 7.1 Die 6 Work Packages (WP 1 – WP 6) für faire Benotung & 6 × 15 Min. Vortrag

Damit jedes der 6 Teammitglieder in seiner **15-minütigen Einzelpräsentation (40 %)** und seinem **Kapitel des KES-Papers (60 %)** mit eigenen wissenschaftlichen Ergebnissen glänzen kann, teilt ihr das Projekt in diese **6 verzahnten Arbeitspakete** auf:

* **Person 1 – WP 1: Data Engineering, Sensor-Biophysik & Cross-Dataset-Harmonisierung (Team Leader)**
  * *Aufgaben:* Beschaffung von `UCDDB`, `CinC 2018`, `ISRUC` (sofort) und `SHHS1/2`, `MESA` (NSRR); Bau der 6-stufen Harmonisierungs-Pipeline (Kanal-Mapping, 0,3–40 Hz Filterung, Multi-Rate-Resampling, Artefakt-Interpolation); experimenteller Vergleich: *Keine Normierung vs. Fensterweiser Z-Score (Alarcón 2023) vs. Robuste Per-Subject IQR-Skalierung (U-Sleep) + Delta-SpO2*.
  * *15-Min.-Vortrag:* Klinischer Hintergrund, Sensorik/Physik aller PSG-/PM-Signale, Heterogenität öffentlicher Datenbanken und quantitative Ergebnisse der Harmonisierungs-Ablation.
* **Person 2 – WP 2: Kardiorespiratorische Apnoe-Detektion & Multiklassen-Erweiterung (PM-Fokus)**
  * *Aufgaben:* Exakte Reproduktion der 1D-CNN-Baseline von **Alarcón et al. (2023)** auf den 4 PM-Signalen (`SpO2`, `HR`, `Thor-Res`, `Abdo-Res`); Untersuchung höherer Abtastraten (`10 Hz` vs. `1 Hz` für Atmung) und Einbezug von `Airflow` / `EDR`; Erweiterung von binärer Klassifikation auf **Multiklassen (`Normal` vs. `OSA` vs. `CSA` vs. `Hypopnoe`)** inkl. Behandlung der Class Imbalance (Focal Loss / Class Weights).
  * *15-Min.-Vortrag:* Portable Monitors vs. PSG, kardiorespiratorische Signalmuster (Bradykardie/Tachykardie, RIP-Paradoxie, SpO2-Totzeit), 1D-CNN/ResNet-Architektur und Vergleich Binär vs. Multiklassen.
* **Person 3 – WP 3: Neurophysiologische EEG-Analyse, Schlafstadien-Scoring & Echte TST-AHI-Berechnung**
  * *Aufgaben:* Verarbeitung der EEG-Kanäle (`C4-A1`, `C3-A2`), Sub-Frame-Analyse (`2–10 s`), Extraktion der 6 Frequenzbänder (`δ, θ, α, σ, β, γ`) & Arousal-Transienten; Bau des **Schlafstadien-/Schlaf-Wach-Moduls** (5 AASM-Stadien W, N1, N2, N3, REM); Berechnung der **wahren Total Sleep Time (TST)** und Nachweis, wie stark sich die AHI-Schätzung (`R²` und Spezifität bei `AHI < 5`) verbessert, wenn man durch `TST` statt durch `TRT` teilt!
  * *15-Min.-Vortrag:* EEG-Biophysik, Frequenzbänder in Hz, Spindeln/K-Komplexe/Arousals, Rechts-Hemisphären-Dominanz (`C4` vs. `C3`) und der quantitative Beweis der TST-korrigierten AHI-Berechnung.
* **Person 4 – WP 4: Multimodale Fusions-Architekturen, Demografie-Integration & Missing-Modality Resilience**
  * *Aufgaben:* Fusion der kardiorespiratorischen Signale (WP 2), der EEG-/Schlafstadien-Information (WP 3) und der **klinisch-demografischen Parameter (Alter, Geschlecht, BMI, Körperlage)**; systematischer Architektur-Vergleich: **Early Fusion** vs. **Intermediate Feature Fusion (Dual-/Multi-Branch mit ECA / Cross-Modal Attention)** vs. **Late Decision Fusion (Stacking)**; Training mit **Modality Dropout**, damit das Modell auch bei fehlenden Sensoren stabil bleibt.
  * *15-Min.-Vortrag:* Theorie der Multimodalität (**Salvi et al., 2024**), Schließen der „Mixed Multimodality“-Forschungslücke im Schlafbereich, Vergleich der 3 Fusionsstrategien und systematische Sensor-Ablationsstudie (welcher Sensor bringt wie viel Prozent Gewinn?).
* **Person 5 – WP 5: Explainable AI (1D-Grad-CAM & SHAP), Uncertainty Quantification & Cross-Dataset-Generalisierung**
  * *Aufgaben:* Implementierung von **Multi-Channel 1D-Grad-CAM** (zeitliche Lokalisierung auf den 60s-Signalkurven) und **SHAP / Channel-Attention-Gewichten** (welcher Sensor war zu wie viel Prozent ausschlaggebend?); Implementierung der **Unsicherheits-Quantifizierung (Confidence Score / Entropie)**; Durchführung der großen **Cross-Dataset-Experimente** (Training auf Datensatz A+B, blinder Test auf ungesehenem Datensatz C).
  * *15-Min.-Vortrag:* Das Black-Box-Problem in der Medizin, Mathematik von 1D-Grad-CAM und SHAP auf Zeitreihen, medizinische Fallstudien (True Positives, False Positives, Grenzfälle) und Cross-Dataset-Generalisierungsergebnisse.
* **Person 6 – WP 6: Lokale, Datenschutzkonforme Software-Architektur, Interaktive GUI & End-to-End-Integration**
  * *Aufgaben:* Entwicklung der **100 % lokalen, DSGVO-konformen Python-Applikation** (EDF/CSV-Importer, schnelle lokale Inferenz via PyTorch/ONNX, interaktiver Multi-Kanal-Signalviewer mit Synchron-Zoom, Hypnogramm- & Event-Zeitstrahl, Grad-CAM-Heatmap-Overlay, automatischer Befundtext-Generator & PDF/HTML-Arztbericht); Laufzeit- und Speicher-Benchmarking auf normaler Laptop-Hardware.
  * *15-Min.-Vortrag:* Datenschutz/DSGVO & Clinical Decision Support (**Secinaro et al., 2021**), Software- & Systemarchitektur, **Live-Demonstration der fertigen Applikation** an einer echten Patientendatei und Gesamtfazit des Projekts.

---

### 7.2 Kompakter 10-Wochen-Zeitplan (Ziel: Komplett fertig bis Ende Dezember 2026!)

* **Woche 1–2 (19. Okt. – 01. Nov. 2026) – Kick-off & Datenbasis:**
  * Rollen (WP 1–6) festlegen, Git-Repo & Zotero-Gruppe einrichten.
  * Sofort-Download von **UCDDB** (`25` PSGs), **ISRUC-Sleep** (`118` PSGs) und **CinC 2018** (Sub-Sample); NSRR-Antrag für **SHHS / MESA** stellen (bzw. Betreuer nach lokalem Labor-Datensatz fragen).
  * Schnittstellen-Definition (einheitliches HDF5-/Zarr-Format zwischen WP 1 und WP 2–6).
* **Woche 3–4 (02. Nov. – 15. Nov. 2026) – Harmonisierungs-Pipeline & Unimodale Baselines:**
  * **WP 1:** 6-Stufen-Preprocessing läuft automatisiert über UCDDB, ISRUC und CinC 2018.
  * **WP 2:** 1D-CNN-Baseline nach *Alarcón et al. (2023)* läuft und liefert erste Metriken.
  * **WP 3:** EEG-Subband-Extraktion (`δ, θ, α, σ, β, γ`) & Schlafstadien-/TST-Klassifikator steht.
  * **WP 6:** GUI-Prototyp lädt `.edf`-Dateien und plottet interaktive Zeitsignale.
* **Woche 5–6 (16. Nov. – 29. Nov. 2026) – Multimodale Fusion, TST-AHI-Kopplung & XAI:**
  * **WP 4:** Kopplung von Kardiorespiratorik + EEG + Demografie (Early vs. Attention vs. Late Fusion + Modality Dropout).
  * **WP 2 & 3:** Berechnung des TST-korrigierten AHI & Multiklassen-Training (Normal / OSA / CSA / Hypopnoe).
  * **WP 5:** 1D-Grad-CAM und SHAP-Erklärungen liefern fertige Arrays für die GUI.
  * **Paper:** Aufsetzen des **KES-Conference-Templates**, Vorschreiben von *Introduction, Related Work* und *Methods*.
* **Woche 7–8 (30. Nov. – 13. Dez. 2026) – Cross-Dataset-Evaluation & Finale App-Integration:**
  * Durchführung aller finalen Trainings- und Cross-Dataset-Läufe sowie der Kanal-Ablationsstudie.
  * Einbindung des besten Modells + XAI-Heatmaps + Klartext-Erklärung in die lokale GUI.
  * Erstellung der **exakt 5 erlaubten Grafiken/Tabellen** für das KES-Paper.
* **Woche 9–10 (14. Dez. – 23./30. Dez. 2026) – Interner Projektabschluss vor Silvester:**
  * Fertigstellung des kompletten **8–10-seitigen KES-Papers** (*Results, Discussion, Conclusion*) inkl. internem Team-Review.
  * Code-Freeze & Verpackung der lokalen Applikation (1-Klick-Start + Beispiel-EDF-Dateien).
  * Erstellung der Präsentationsfolien (6 × 15 Min.).
* **Januar 2027 (Puffer-Monat):**
  * Einarbeiten von Betreuer-Feedback in Paper und Folien; 2 Generalproben des 90-Minuten-Vortrags mit Stoppuhr.

---

### 7.3 Euer Masterplan für die exakt 5 erlaubten Grafiken/Tabellen im KES-Paper
Da das KES-Paper **maximal 5 grafische Elemente (Tabellen + Abbildungen zusammen!)** enthalten darf, nutzt ihr diese 5 Plätze wie folgt:
1. **Figure 1 (System- & Fusions-Architektur):** Kompaktes Blockdiagramm eurer End-to-End-Pipeline (6-Stufen-Harmonisierung -> Multi-Branch Feature Extraction -> Attention Fusion + TST-Kopplung -> XAI & Lokale GUI).
2. **Table 1 (Datensatz-Übersicht & Ablationsstudie):** Kombinierte Tabelle der genutzten Datensätze (UCDDB, CinC 2018, SHHS2, MESA) UND der Ablationsergebnisse (Fensterweiser Z-Score vs. Robuste IQR-Skalierung; Unimodal vs. Early vs. Attention vs. Late Fusion).
3. **Figure 2 (Multi-Panel Evaluierungs-Grafik):**
   * *Links:* ROC-Kurven (In-Dataset & Cross-Dataset AUC).
   * *Mitte:* Konfusionsmatrix für Normal / Hypopnoe / OSA / CSA.
   * *Rechts:* AHI-Scatterplot (`R²`-Vergleich: AHI über TRT [Alarcón 2023] vs. AHI über echte TST).
4. **Figure 3 (XAI-Fallstudie – 1D-Grad-CAM + Sensor-Attribution):** Darstellung eines echten 60s-Apnoe-Fensters (`Airflow`, `Thor-Res`, `Abdo-Res`, `SpO2`, `HR`, `EEG C4-A1`) mit hinterlegter Grad-CAM-Heatmap und prozentualem Sensor-Beitrag.
5. **Table 2 (State-of-the-Art-Vergleich & Lokale Laufzeit):** Direkter Vergleich eurer Ergebnisse mit *Alarcón et al. (2023)*, *Peng et al. (2024/2025)*, *Huttunen et al. (2022)* und *Gutiérrez-Tobal et al. (2021)* hinsichtlich Accuracy, Sensitivity, Specificity, F1, AUC, AHI-`R²`, XAI-Unterstützung und lokaler Rechenzeit pro 8h-Aufnahme.


---

## 8. Medizinisches & Theoretisches Tiefen-Kompendium: Schlafphysiologie, Schlafphasen, Schlafapnoe, Andere Schlafstörungen (ICSD-3) & Evaluations-Metriken

Dieses Kapitel liefert euch das vollständige medizinische, physiologische und methodische Fundament für **Phase 1 & 2 eures Projekts** (*„Acquire basic knowledge on sleep, sleep disorders, sleep apnea detection, multimodality and AI evaluation“*). Es ist bewusst so aufgebaut, dass ihr sowohl für **Schlafapnoe** als auch für **jede andere Schlafanalyse** (Schlafstadien, Insomnie, PLMS, REM-Schlaf-Verhaltensstörung, Schlafqualität) alle klinischen Definitionen, Formeln und Signal-Biomarker sofort parat habt.

---

### 8.1 Physiologie des Schlafs, Zwei-Prozess-Modell & Das Hypnogramm

#### A. Warum und wie wir schlafen: Das Zwei-Prozess-Modell der Schlafregulation (Borbély, 1982)
Der menschliche Schlaf-Wach-Rhythmus wird durch das Zusammenspiel zweier unabhängiger biologischer Prozesse gesteuert:
1. **Prozess S (Homöostatischer Schlafdruck):**
   * Während jeder Wachstunde steigt im Gehirn (insbesondere im basalen Vorderhirn) die Konzentration von **Adenosin** als Abbauprodukt des zellulären Energieverbrauchs (ATP-Abbau) kontinuierlich an.
   * Je länger man wach ist, desto höher wird dieser homöostatische Schlafdruck *(Hinweis: Koffein wirkt als kompetitiver Antagonist exakt an diesen Adenosin-A1/A2A-Rezeptoren!)*.
   * Sobald wir einschlafen, wird Prozess S exponentiell abgebaut. **Direkter Biomarker im EEG:** Die Stärke des homöostatischen Schlafdrucks spiegelt sich Eins-zu-eins in der **Slow-Wave Activity (SWA, spektrale Leistungsdichte im Delta-Band `0,5–4,0 Hz`)** des Tiefschlafs (Stadium N3) wider. Deshalb tritt der meiste Tiefschlaf in den ersten 2–3 Stunden der Nacht auf!
2. **Prozess C (Zirkadianer Rhythmus / Innere 24-Stunden-Uhr):**
   * Gesteuert durch den **Nucleus suprachiasmaticus (SCN)** im Hypothalamus, synchronisiert durch blauhaltiges Tageslicht über die melanopsinhaltigen Ganglienzellen der Netzhaut.
   * Steuert die Ausschüttung von **Melatonin** (aus der Zirbeldrüse bei Dunkelheit -> fördert das Einschlafen), **Cortisol** (Minimum um Mitternacht, starker Anstieg in den frühen Morgenstunden -> *Cortisol Awakening Response*) und die **Körperkerntemperatur** (fällt beim Einschlafen um ca. `0,5–1,0 °C` ab, erreicht ihr Minimum gegen `03:00–05:00 Uhr` morgens – dort ist die Neigung zum **REM-Schlaf** am höchsten!).

#### B. Der Ultradiane Schlafzyklus (Die 90-Minuten-Architektur der Nacht)
Ein gesunder Erwachsener durchläuft pro Nacht **4 bis 6 Schlafzyklen** von jeweils ca. **`90 bis 110 Minuten` Dauer**:
* **Reihenfolge innerhalb eines Zyklus:** `Wake (W) -> N1 -> N2 -> N3 -> N2 -> REM (R)`.
* **Wichtige Verschiebung über die Nacht:**
  * **1. Nachthälfte (Zyklus 1 & 2):** Dominiert von langem **Tiefschlaf (Stadium N3)** (hoher homöostatischer Druck Prozess S), während die REM-Phasen noch sehr kurz sind (`5–15 Min.`).
  * **2. Nachthälfte (Zyklus 3 bis 5/6):** Prozess S ist abgebaut -> **Stadium N3 verschwindet fast vollständig**, stattdessen wechseln sich **Stadium N2** und immer länger werdende **REM-Schlaf-Phasen (`30–60 Min.`)** ab.
  * **Konsequenz für die Schlafapnoe:** Da obstruktive Apnoen wegen der kompletten Muskelerschlaffung (Atonie) im **REM-Schlaf** am häufigsten und schwersten sind, häufen sich die längsten Apnoen und tiefsten Sauerstoffabfälle (`SpO2 < 80 %`) oft in den **frühen Morgenstunden (zwischen 03:00 und 06:00 Uhr)**!

```text
TYPISCHES HYPNOGRAMM EINES GESUNDEN ERWACHSENEN vs. SCHLAFAPNOE-PATIENT (8 STUNDEN)

Gesunder Schlaf (Klare 90-Min.-Zyklen, viel N3 am Anfang, lange REM-Blöcke am Morgen):
W   |===                                      |                     |              ===
REM |            ===          ======          |      =========      |   ===========
N1  |   ==     ==   ==      ==      ==        |    ==         ==    | ==
N2  |     ===         ====            ======  |====             ====|
N3  |        ====         ====                |                     |
    +-----------------------------------------+---------------------+----------------->
    23:00      00:30        02:00           03:30                 05:00            07:00

Schwere Schlafapnoe (Massive Schlaf-Fragmentierung durch ständige Arousals, kaum N3/REM!):
W   |===  |  |   |  |   |   |  |   |  |   |   |  |   |   |  |   |   |  |   |  |    ===
REM |                                         |      =   =          |      =  =
N1  |   == == === == === === == === == === ===|== === === == === ===|== === == ====
N2  |     =  =   =  =   =   =  =   =  =   =   |  =   =   =  =   =   |  =   =  =
N3  |        =                                |                     |
    +-----------------------------------------+---------------------+----------------->
    23:00      00:30        02:00           03:30                 05:00            07:00
```

#### C. Alle klinischen Schlafarchitektur-Kennzahlen (Formeln für euer Programm!)
Wenn euer Programm die Schlafphasen pro 30-Sekunden-Epoche bestimmt, könnt ihr sofort folgende klinische Standard-Metriken berechnen:
1. **TIB (Time in Bed) / TRT (Total Recording Time):** Gesamtdauer der Aufzeichnung vom „Licht löschen“ (*Lights Off*) bis „Licht an“ (*Lights On*) in Minuten bzw. Stunden (`N_total_epochs * 0,5 Min.`).
2. **TST (Total Sleep Time – Wahre Gesamtschlafzeit):** Summe aller als Schlaf klassifizierten Epochen (`N1 + N2 + N3 + REM`):
   `TST [Minuten] = (Anzahl Epochen in N1, N2, N3, REM) * 0,5`
   `TST [Stunden] = TST [Minuten] / 60`
3. **SOL (Sleep Onset Latency – Einschlaflatenz):** Zeit vom Aufzeichnungsbeginn (*Lights Off*) bis zur **allerersten Schlaf-Epoche** (egal welches Schlafstadium). *Normalwert:* `10–20 Min.` (`> 30 Min.` spricht für Einschlaf-Insomnie; `< 8 Min.` spricht für exzessive Tagesschläfrigkeit / Narkolepsie).
4. **WASO (Wake After Sleep Onset):** Summe aller **Wach-Epochen (W)**, die *nach* dem ersten Einschlafen und *vor* dem finalen Aufwachen am Morgen auftreten. *Normalwert:* `< 30–40 Min.` (bei Schlafapnoe und Durchschlaf-Insomnie stark erhöht).
5. **REM-Latenz (REM Onset Latency):** Zeit vom ersten Einschlafen bis zur **ersten REM-Epoche**. *Normalwert:* `70–120 Min.`
   * **Klinischer Spezialfall SOREMP (*Sleep-Onset REM Period*):** Tritt REM-Schlaf bereits innerhalb von **`<= 15 Minuten`** nach dem Einschlafen auf, ist dies das Kardinalsymptom für **Narkolepsie**!
6. **SE (Sleep Efficiency – Schlafeffizienz):**
   `SE [%] = (TST / TIB) * 100 %`
   *Normalwert:* **`>= 85 %`** (`< 80 %` gilt als klinisch gestörter Schlaf).
7. **Prozentuale Stadien-Verteilung (relativ zur `TST`) & Normwerte beim Erwachsenen:**
   * `% N1 = (Zeit in N1 / TST) * 100 %` -> *Norm:* **`2 % – 5 %`** (bei Schlafapnoe oft auf `15–30 %` erhöht, da Patienten nach jedem Arousal wieder in N1 zurückfallen!).
   * `% N2 = (Zeit in N2 / TST) * 100 %` -> *Norm:* **`45 % – 55 %`**.
   * `% N3 = (Zeit in N3 / TST) * 100 %` -> *Norm:* **`15 % – 25 %`** (nimmt mit steigendem Lebensalter physiologisch ab; bei schwerer OSA oft `< 5 %`!).
   * `% REM = (Zeit in REM / TST) * 100 %` -> *Norm:* **`20 % – 25 %`**.

---

### 8.2 Die 5 Schlafphasen im Detail: R&K (1968) vs. AASM, Physiologie & Übergangs-Logik

#### A. Historische Entwicklung: Rechtschaffen & Kales (1968) vs. AASM-Standard (ab 2007 bis heute)
Da ältere öffentliche Datenbanken (wie *UCDDB*, *Sleep-EDF*, *MIT-BIH*, *CAP*) noch nach dem **R&K-Standard (1968)** annotiert wurden, neuere (wie *CinC 2018*, *ISRUC*, *SHHS-AASM*) jedoch nach **AASM**, muss jedes Teammitglied den exakten Unterschied kennen:

| Merkmal | Rechtschaffen & Kales (R&K, 1968) | AASM Manual (Version 2.6 / 3.0) | Umrechnungsregel für eure Daten-Pipeline |
| :--- | :--- | :--- | :--- |
| **Anzahl der Stadien** | **6 Schlafstadien + Movement Time (MT)**:<br>`W`, `S1`, `S2`, `S3`, `S4`, `REM`, `MT` | **5 Stadien**:<br>`W` (Wake), `N1`, `N2`, `N3`, `R` (REM) | • `S1 -> N1`<br>• `S2 -> N2`<br>• **`S3 + S4 -> N3`** (Zusammenlegung!)<br>• `REM -> R`<br>• `MT` -> verwerfen oder der vorangehenden/Wach-Klasse zuordnen |
| **Tiefschlaf-Grenze** | `S3`: `20 % – 50 %` Delta-Wellen pro Epoche<br>`S4`: `> 50 %` Delta-Wellen pro Epoche | **`N3`**: Sobald **`>= 20 %`** einer 30s-Epoche (`>= 6 s`) Delta-Wellen (`0,5–2 Hz`, `> 75 µV` frontal) zeigen | Keine Trennung zwischen S3 und S4 mehr nötig, da physiologisch identisch. |
| **Empfohlene EEG-Kanäle** | Nur **1 zentraler Kanal** Pflicht (`C3-A2` oder `C4-A1`) | **3 Kanäle aus 3 Regionen** Pflicht:<br>Frontal (`F4-M1`), Zentral (`C4-M1`), Okzipital (`O2-M1`) | Falls ein Datensatz alle 3 Regionen hat: nutze `F4` (Delta/K-Komplex), `C4` (Spindeln/Apnoe) und `O2` (Alpha/Arousal); falls nur 1 Kanal gewählt wird: **`C4-M1` (`C4-A1`)**. |

#### B. Ganzkörper-Physiologie über die 5 Schlafstadien hinweg

| Physiologisches System | Wach (`W`) | Leichtschlaf (`N1` & `N2`) | Tiefschlaf (`N3` / Slow-Wave Sleep) | Traumschlaf (`R` / REM-Schlaf) |
| :--- | :--- | :--- | :--- | :--- |
| **Gehirn (EEG)** | Alpha (`8–12 Hz` okzipital bei geschlossenen Augen) & Beta (`13–30 Hz`) | **N1:** Theta (`4–7 Hz`), Vertex-Wellen<br>**N2:** **Schlafspindeln (`11–16 Hz`)** & **K-Komplexe (`>= 0,5 s`)** | Hochamplitudige **Delta-Wellen (`0,5–4 Hz`, `> 75 µV`)**, höchste kortikale Synchronisation | Desynchronisiertes Theta/Beta + **Sägezahnwellen (`2–6 Hz`)**; Gehirnstoffwechsel fast so hoch wie im Wachzustand! |
| **Augen (EOG)** | Schnelle Blicksprünge & Blinzeln (`0,5–2 Hz`) | **N1:** Langsame rollende Augenbewegungen (**SEMs**, `0,2–0,5 Hz`)<br>**N2:** Keine Augenbewegungen | Keine Augenbewegungen (EOG zeigt nur eingestreute Stirn-Delta-Wellen) | **Phasische Bursts schneller, sakkadischer Augenbewegungen (REMs)** |
| **Muskeltonus (Kinn-EMG)** | Hoch und variabel | Moderat (niedriger als im Wachzustand) | Niedrig (weiter abgesenkt) | **Komplette Skelettmuskel-Atonie (Muskellähmung bis auf Zwerchfell & Augenmuskeln!)** |
| **Herz & Kreislauf (EKG / Blutdruck)** | Hoch, variabel (Sympathikus-dominant) | Absinken von Herzfrequenz und Blutdruck; **Parasympathikus (Vagus)** übernimmt | **Tiefster Punkt von Herzfrequenz und Blutdruck („Nächtliches Dipping“ um 10–20 %)**, extrem regelmäßige R-R-Intervalle (hohe HF-HRV) | **Vegetativer Sturm (Instabilität):** Herzfrequenz und Blutdruck schwanken stark zwischen Bradykardie und Tachykardie (Sympathikus-Bursts) |
| **Atmung & Obere Atemwege** | Willkürlich + metabolisch gesteuert; hoher Tonus der Rachenmuskulatur | Rein metabolisch (CO2-gesteuert); im Übergang W->N1 oft leicht periodisch; Rachentonus sinkt leicht | **Stabilste, gleichmäßigste Atmung der ganzen Nacht!** Zentrale Chemoreflex-Steuerung ist ruhig -> **wenigste Apnoen in N3!** | Unregelmäßige Atemfrequenz & Atemzugtiefe; **Tonus des Zungenmuskels (*M. genioglossus*) fällt auf nahezu Null** -> **längste und schwerste obstruktive Apnoen in REM!** |

#### C. Biologische Übergangs-Grammatik zwischen Schlafphasen (Wichtig für Sequenzmodelle!)
Schlafstadien wechseln nicht völlig zufällig von einer 30s-Epoche zur nächsten, sondern folgen biologischen Regeln:
* **Hohe Selbst-Übergangswahrscheinlichkeit:** Wenn Epoche `t` in `N2`, `N3` oder `REM` ist, liegt die Wahrscheinlichkeit, dass Epoche `t+1` im selben Stadium bleibt, bei `> 85–92 %`.
* **Biologisch unplausible Sprünge:** Ein gesunder Mensch springt niemals direkt von `Wake (W)` in `Tiefschlaf (N3)` oder direkt von `Tiefschlaf (N3)` in `REM` (dazwischen liegt fast immer `N1`/`N2`). Umgekehrt kann man aus **jedem** Schlafstadium (`N1, N2, N3, REM`) durch ein Arousal sofort nach **`Wake (W)`** oder **`N1`** springen!
* **Technischer Nutzen:** Genau deshalb erzielen Modelle, die **Nachbar-Epochen mitbetrachten** (wie *BiLSTM*, *GRU*, *Transformer* oder eine *Conditional Random Field [CRF] / Hidden Markov Model [HMM]* Nachbearbeitungsschicht), um **`+4 % bis +8 %` höhere Genauigkeiten** als reine Einzel-Epochen-CNNs!

---

### 8.3 Schlafapnoe & Schlafbezogene Atmungsstörungen (SBAS) – Pathophysiologie, Subtypen & Indizes

#### A. Der Pathomechanismus der Obstruktiven Schlafapnoe (OSA) – „Der nächtliche Teufelskreis“
Weltweit sind ca. **936 Millionen bis 1 Milliarde Erwachsene** zwischen 30 und 69 Jahren von obstruktiver Schlafapnoe betroffen (über 80 % davon unentdeckt!). Was passiert dabei anatomisch und physiologisch im Körper?

```text
   [1. Einschlafen (N1/N2/REM)]
   Tonus der Rachen-Dilatatoren (v.a. M. genioglossus / Zungengrund) lässt nach
                               |
                               v
   [2. Einatmung (Inspiration) -> Bernoulli-Effekt]
   Zwerchfell erzeugt Unterdruck -> verengter Pharynx kollabiert komplett (P_crit überschritten)
                               |
                               v
   [3. Obstruktive Apnoe (>= 10 s bis > 60 s)]
   • Oronasaler Luftstrom = 0 (Stille am Mikrofon)
   • Frustrane, paradoxe Atemanstrengung (Brust zieht ein, Bauch wölbt vor; Druck bis -60 cmH2O!)
                               |
                               v
   [4. Blutgas-Entgleisung & Kardiale Reaktion]
   • Arterielle Hypoxie (SpO2 fällt um >= 3% bis < 70%) + Hyperkapnie (pCO2 steigt) + Azidose
   • Vagus-Tauchreflex -> Progressive Sinusbradykardie (Herzfrequenz fällt z.B. auf 45 bpm)
                               |
                               v
   [5. Notfall-Weckreaktion (Kortikales & Autonomes Arousal)]
   • Chemorezeptoren & Mechanorezeptoren schlagen im Hirnstamm Alarm
   • EEG springt für >= 3 s in Alpha/Beta-Frequenz; Kinn-EMG feuert -> Rachen öffnet sich explosiv!
   • Sympathikus-Sturm: Periphere Vasokonstriktion (PPG-Drop), Blutdruck-Spitze (> 200 mmHg),
     Herzfrequenz schießt hoch (Tachykardie, z.B. 85 bpm), lautes Post-Apnoe-Luftschnappen (PAS)
                               |
                               v
   [6. Hyperventilation & Wiedereinschlafen]
   • 3 bis 5 tiefe Atemzüge normalisieren SpO2 und waschen CO2 aus
   • Patient schläft sofort wieder ein (erinnert sich morgens NICHT an das Aufwachen!)
   • Zyklus beginnt sofort von vorn (bis zu 400-mal pro Nacht!)
```

#### B. Exakte Abgrenzung aller 5 respiratorischen Event-Typen nach AASM (v2.6 / v3.0)

| Event-Typ | Primärer AASM-Messsensor | Exakte quantitative Scoring-Regel (Erwachsene) | Verhalten von Thorax- & Abdomen-Gurt (`Thor-Res` / `Abdo-Res`) | Pathophysiologische Ursache |
| :--- | :--- | :--- | :--- | :--- |
| **1. Obstruktive Apnoe (OSA)** | Oronasaler Thermistor *(Alternativ: Nasaldruck)* | • Abfall der Peak-Signalamplitude um **`>= 90 %`** gegenüber der Prä-Event-Baseline<br>• Dauer **`>= 10 Sekunden`**<br>• *(Kein SpO2-Abfall für die Wertung als Apnoe erforderlich!)* | **Fortgesetzte oder zunehmende Atemanstrengung während der gesamten Dauer** des Atemstillstands (meist **gegenphasig / paradox** zwischen Brust und Bauch). | Mechanischer Verschluss der oberen Atemwege (Pharynxkollaps) bei intaktem zentralem Atemantrieb. |
| **2. Zentrale Apnoe (CSA)** | Oronasaler Thermistor + RIP-Gurte | • Abfall der Thermistor-Amplitude um **`>= 90 %`** gegenüber Baseline<br>• Dauer **`>= 10 Sekunden`** | **Komplettes Fehlen jeglicher inspiratorischer Anstrengung** (sowohl `Thor-Res` als auch `Abdo-Res` sind während des gesamten Events eine flache Nulllinie!). | Temporärer Ausfall des Atemzentrums im Hirnstamm (Pontomedullärer Schrittmacher sendet keinen Impuls an das Zwerchfell). Oft bei Herzinsuffizienz (*Cheyne-Stokes-Atmung*), Schlaganfall oder Opioiden. |
| **3. Gemischte Apnoe (MSA)** | Oronasaler Thermistor + RIP-Gurte | • Abfall der Thermistor-Amplitude um **`>= 90 %`** gegenüber Baseline<br>• Dauer **`>= 10 Sekunden`** | **Zweigeteiltes Muster:** Im **ersten Abschnitt** fehlt jede Atemanstrengung (wie CSA, flache Gurte), im **zweiten Abschnitt** setzt vor Wiederbeginn des Luftstroms die Atemanstrengung wieder ein (wie OSA), während der Rachen noch verschlossen ist. | Epiglottis/Pharynx fällt während einer initialen zentralen Atempause passiv zu und blockiert dann den wieder einsetzenden Atemversuch. |
| **4. Hypopnoe (Obstruktiv / Zentral)** | **Nasaldruck-Kanüle (NPT)** + Pulsoximeter + EEG | • Abfall der Nasaldruck-Amplitude um **`>= 30 %`** gegenüber Baseline<br>• Dauer **`>= 10 Sekunden`**<br>• **PLUS zwingend:**<br>  - Entweder **`>= 3 %` SpO2-Desaturierung** ODER ein **EEG-Arousal** *(AASM 1A – Empfohlen)*<br>  - *Alternativ (AASM 1B):* **`>= 4 %` SpO2-Desaturierung** | • **Obstruktive Hypopnoe:** Schnarchen während des Events und/oder Abflachung der inspiratorischen Nasaldruckkurve und/oder thorako-abdominale Paradoxie.<br>• **Zentrale Hypopnoe:** Keines dieser 3 Obstruktionszeichen vorhanden (synchrone, flachere Atmung). | Partielle Verengung der oberen Atemwege um 30–89 % mit relevanter Sauerstoffentsättigung oder Schlaffragmentierung. |
| **5. RERA (*Respiratory Effort-Related Arousal*)** | Nasaldruck-Kanüle (NPT) + EEG | • Sequenz von Atemzügen über **`>= 10 Sekunden`**, die *nicht* die Kriterien einer Apnoe oder Hypopnoe erfüllt, aber durch **zunehmende Atemanstrengung oder Abflachung der inspiratorischen Druckkurve (*Flow Limitation*)** gekennzeichnet ist<br>• **UND in einem EEG-Arousal endet!** | Zunehmende Druck-/Gurtanstrengung ohne `>= 30 %` Flussabfall und ohne `>= 3 %` SpO2-Drop, abgebrochen durch ein Weck-Ereignis im EEG. | Erhöhter Atemwegswiderstand (*Upper Airway Resistance Syndrome, UARS*), der bereits vor einer echten Hypoxie das Gehirn aufweckt. |

> **WICHTIGER HINWEIS ZU KINDERN (Pädiatrische AASM-Regeln, z. B. im CHAT-Datensatz!):**
> Da Kinder eine höhere Atemfrequenz und eine geringere funktionelle Residualkapazität der Lunge haben, gilt bei Kindern (`< 18 Jahre`) eine **Obstruktive Apnoe oder Hypopnoe bereits ab einer Dauer von `>= 2 Atemzügen` (ca. `4–6 Sekunden` statt `10 Sekunden`!)**, und bereits ein **`AHI >= 1 Event/Stunde`** gilt bei Kindern als pathologisch (`1–5`: mild, `5–10`: moderat, `>= 10`: schwer)!

---

#### C. Alle klinischen Schweregrad-Indizes & Biomarker (Formeln für euer Analyse-Modul)

1. **AHI (Apnea-Hypopnea Index – Goldstandard mit bekannter Schlafzeit `TST`):**
   `AHI [Events/h] = (Anzahl aller Apnoen [OSA + CSA + MSA] + Anzahl aller Hypopnoen) / Total Sleep Time (TST in Stunden)`
2. **REI (Respiratory Event Index – Standard für Portable Monitors ohne Schlafzeit-Erkennung):**
   `REI [Events/h] = (Anzahl aller Apnoen + Hypopnoen) / Total Recording Time (TRT in Stunden)`
   *(Hinweis: Da `TRT >= TST` ist, gilt mathematisch bei fehlerfreier Event-Zählung immer `REI <= AHI`. Genau deshalb ist euer Dual-Modul, das `TST` mitschätzt, klinisch so wertvoll!)*
3. **RDI (Respiratory Disturbance Index – inkl. RERAs):**
   `RDI [Events/h] = (Anzahl Apnoen + Anzahl Hypopnoen + Anzahl RERAs) / TST [Stunden]`
4. **Klinische Schweregrad-Klassifikation bei Erwachsenen (nach AASM):**
   * **Normal / Physiologisch:** `AHI < 5,0 Events/h`
   * **Leichte (Milde) Schlafapnoe:** `5,0 <= AHI < 15,0 Events/h`
   * **Mittelschwere (Moderate) Schlafapnoe:** `15,0 <= AHI < 30,0 Events/h`
   * **Schwere (Severe) Schlafapnoe:** `AHI >= 30,0 Events/h`
5. **Ergänzende Oximetrie- & Phänotyp-Metriken (machen euer Programm zu einem echten Profi-Tool!):**
   * **ODI3% / ODI4% (Oxygen Desaturation Index):** Anzahl der SpO2-Abfälle um `>= 3 %` bzw. `>= 4 %` pro Stunde Schlafzeit.
   * **T90 (Time below 90 % SpO2):** Prozentualer Anteil der Schlafzeit, in der die Sauerstoffsättigung unter `< 90 %` lag (starker Indikator für chronische Hypoxie).
   * **MAD (Mean Apnea-Hypopnea Duration):** Durchschnittliche Dauer eines respiratorischen Events in Sekunden (ein Patient mit `AHI = 20` und `45 s` langen Apnoen ist klinisch viel stärker gefährdet als ein Patient mit `AHI = 20` und `11 s` kurzen Apnoen!).
   * **Hypoxic Burden (`% * min / h`):** Fläche zwischen der SpO2-Baseline und der SpO2-Kurve während aller Events, geteilt durch die Schlafzeit in Stunden. Gilt in der aktuellen Schlafforschung (2022–2026) als bester Prädiktor für Herzinfarkt- und Schlaganfallrisiko!
   * **REM-AHI vs. NREM-AHI:** Separate Berechnung des AHI im REM-Schlaf (`AHI_REM`) und im NREM-Schlaf (`AHI_NREM`). Ist `AHI_REM / AHI_NREM >= 2,0`, liegt eine **REM-prädominante OSA** vor.
   * **Supine-AHI vs. Non-Supine-AHI:** Ist `AHI_Rückenlage / AHI_Seitenlage >= 2,0`, liegt eine **lageabhängige OSA (Positional OSA)** vor, die oft schon durch eine einfache Rückenlage-Verhinderungsweste therapiert werden kann!

---

### 8.4 Weitere relevante Schlafstörungen nach ICSD-3 (Falls ihr das Projekt breiter aufstellt!)

Da du gesagt hast, dass es eventuell auch eine **allgemeinere Schlafanalyse / Klassifikation verschiedener Schlafkrankheiten** werden könnte, findest du hier die komplette Übersicht der **6 Hauptkategorien nach ICSD-3 (*International Classification of Sleep Disorders*)** und **woran euer ML-Modell jede dieser Krankheiten in den Sensordaten erkennt**:

| ICSD-3 Krankheitsgruppe | Wichtigste Krankheitsbilder | Primäre diagnostische Sensoren in der PSG | Charakteristische Biomarker & Signal-Signatur für euren Classifier | Öffentliche Datensätze mit diesen Labels |
| :--- | :--- | :--- | :--- | :--- |
| **1. Insomnien** | Chronische Ein- und Durchschlafstörung (*Insomnia Disorder*) | EEG, EOG, Kinn-EMG, Aktigraphie | • **Makroarchitektur:** Einschlaflatenz `SOL > 30 Min.`, Wachzeit nach Einschlafen `WASO > 30–45 Min.`, Schlafeffizienz `SE < 85 %`, verkürzte `TST < 6 h`, reduzierte N3-Dauer.<br>• **Mikroarchitektur (EEG-Hyperarousal):** Abnorm erhöhte relative **Beta- (`15–30 Hz`) und Gamma-Power (`30–45 Hz`) während des NREM-Schlafs** (kortikale Übererregung verhindert tiefes Abschalten) sowie erhöhte Herzfrequenz und reduzierte HF-HRV in der Nacht. | **CAP Sleep Database** (`ins` Subgruppe), **Sleep-EDF** (Subgruppe `ST` mit Einschlafstörungen), **ISRUC-Sleep** |
| **2. Schlafbezogene Atmungsstörungen (SBAS / SDB)** | **OSA**, **CSA**, **MSA**, Cheyne-Stokes-Atmung, Schlafbezogene Hypoventilation | Thermistor, Nasaldruck, RIP-Gurte (`Thor/Abdo`), `SpO2`, EKG, EEG, Mikrofon | • `>= 90 %` (Apnoe) bzw. `>= 30 %` (Hypopnoe) Atemfluss-Reduktion für `>= 10 s`<br>• Paradoxe RIP-Gurtbewegung (OSA) vs. flache Gurte (CSA)<br>• Zyklische SpO2-Entsättigungen (`ODI >= 5/h`)<br>• CVHR im EKG (VLF-Peak bei `0,01–0,03 Hz`) + EEG-Arousals (`> 3 s` Alpha/Beta). | **UCDDB**, **CinC 2018**, **SHHS1/2**, **MESA**, **ISRUC**, **PSG-Audio**, **Apnea-ECG**, **MIT-BIH** |
| **3. Zentrale Störungen mit Tagesschläfrigkeit (Hypersomnien)** | **Narkolepsie Typ 1** (mit Kataplexie / Orexin-Mangel) & **Typ 2**, Idiopathische Hypersomnie | EEG, EOG, Kinn-EMG (Nacht-PSG + Tagestest MSLT) | • **Extrem kurze Einschlaflatenz (`SOL < 8 Min.`)**<br>• **SOREMP (*Sleep-Onset REM Period*):** Übergang von Wach/N1 direkt in den **REM-Schlaf innerhalb von `<= 15 Minuten`** nach dem Einschlafen!<br>•Stark fragmentierter Nachtschlaf mit häufigen kurzen Wach-Episoden und dissoziierten REM-Merkmalen. | **CAP Sleep Database** (`narco` Subgruppe), **MNC (Stanford/NSRR Narcolepsy Cohort)** |
| **4. Schlafbezogene Bewegungsstörungen** | **PLMD (*Periodic Limb Movement Disorder*)**, **RLS (*Restless Legs Syndrome*)**, **Schlaf-Bruxismus** (Zähneknirschen) | **Tibialis-EMG** (Beine), **Kinn-/Masseter-EMG** (Kiefer), EEG, EKG/HR | • **PLMS (Beinbewegungen):** Repetitive Bursts im Schienbein-EMG von `0,5–10 s` Dauer (Anstieg `>= 8 µV` über Ruhe-EMG), mindestens `4` Bursts in Folge im typischen Intervall von **`5 bis 90 Sekunden`** (oft alle `20–40 s` in N1/N2), begleitet von kurzen Herzfrequenz-Anstiegen und EEG-Arousals. Pathologisch ab **`PLMI > 15 Events/h`**.<br>• **Bruxismus:** Phasisches (`>= 3 Bursts` à `0,25–2 s`) oder tonisches (`> 2 s`) hochamplitudiges Knirschen im Kinn-/Masseter-EMG + Audio. | **CAP Sleep Database** (`plm` & `brux` Subgruppen), **ISRUC-Sleep**, **SHHS / MrOS** |
| **5. Parasomnien** | • **NREM-Parasomnien:** Schlafwandeln, Pavor nocturnus<br>• **REM-Parasomnie:** **RBD (*REM Sleep Behavior Disorder*)** | EEG (`F4, C4, O2`), EOG, **Kinn-EMG & Tibialis-EMG**, Video | • **NREM-Parasomnie:** Abruptes Erwachen direkt aus dem **Tiefschlaf (N3)** im ersten Nachtdrittel; das EEG zeigt weiterhin **hochamplitudige langsame Delta-Wellen (*Hypersynchronous Delta*)** gemischt mit Alpha/Beta, während EMG und Herzfrequenz extrem ansteigen.<br>• **RBD (REM-Schlaf-Verhaltensstörung):** **Verlust der physiologischen Muskelatonie im REM-Schlaf (*RSWA – REM Sleep Without Atonia*)**! Während EOG und EEG eindeutigen REM-Schlaf zeigen, weist das Kinn-/Bein-EMG hohe tonische oder phasische Muskelaktivität auf (Patienten schlagen/sprechen im Traum; wichtigster Frühindikator für **Parkinson**!). | **CAP Sleep Database** (`rbd` [22 Patienten!] & `nfle` [Nächtliche Frontallappen-Epilepsie, 40 Patienten!]) |
| **6. Zirkadiane Schlaf-Wach-Rhythmusstörungen** | Verzögerter (*DSWPD*) oder Vorverlagerter (*ASWPD*) Schlafphasentyp, Schichtarbeitersyndrom | Mehr-Tages-Aktigraphie, Hauttemperatur, Lichtsensor, PPG/HRV | • Phasenverschiebung des Aktivitäts-Ruhe-Rhythmus (*Cosinor-Analyse: Mesor, Amplitude, Acrophase*) und des nächtlichen Herzfrequenz-Minimums um `> 2–4 Stunden` gegenüber der sozialen Normzeit.<br>• Normale interne Schlafarchitektur, sobald der Patient zu seiner biologischen Eigenzeit schlafen darf. | **MESA Actigraphy** (7 Tage Handgelenks-Aktigraphie parallel zur PSG!), **STAGES** |

> **GOLDWERT-TIPP FÜR EUER TEAM (Der „CAP Sleep Database“-Trick auf PhysioNet):**
> Falls ihr oder eure Betreuer im Projekt nicht *nur* Schlafapnoe, sondern einen **allgemeinen Multi-Disorder-Schlafklassifikator** bauen wollt, gibt es auf PhysioNet die **CAP Sleep Database (`https://physionet.org/content/capslpdb/1.0.0/`, 108 komplette PSGs, sofort frei herunterladbar!)**. Sie enthält explizit gelabelte Patienten aus **7 verschiedenen Krankheitsgruppen**:
> 1. Gesunde Kontrollen (`n`, 16)
> 2. Schlafbezogene Atmungsstörungen / Schlafapnoe (`sdb`, 4)
> 3. Nächtliche Frontallappen-Epilepsie (`nfle`, 40)
> 4. REM Sleep Behavior Disorder (`rbd`, 22)
> 5. Periodic Limb Movement Disorder (`plm`, 10)
> 6. Insomnie (`ins`, 9)
> 7. Narkolepsie (`narco`, 5) & Bruxismus (`brux`, 2)
> Damit könntet ihr bei Bedarf sogar zeigen, dass eure Pipeline neben Schlafapnoe auch Insomnie, PLMD oder RBD an ihren charakteristischen EEG-/EMG-/Atmungs-Mustern unterscheiden kann!

---

### 8.5 Weitere relevante klinische & methodische Konzepte für das KES-Paper

#### 1. Die 4 klinischen Geräteklassen der Schlafdiagnostik (AASM Level I bis Level IV)
In der Einleitung und Diskussion eures Papers solltet ihr präzise benennen, in welche medizinische Gerätekategorie euer Modell fällt (siehe Alarcón et al., 2023 & Peng et al., 2025):
* **Level I (Überwachte Labor-Polysomnographie / Standard-PSG):** Mindestens **7 Kanäle** (EEG, EOG, Kinn-EMG, EKG, Airflow, Atmungs-Effort, `SpO2`), aufgezeichnet im Schlaflabor unter ständiger Überwachung durch geschultes Personal. *Der klinische Goldstandard.*
* **Level II (Unüberwachte Voll-Polysomnographie zu Hause):** Mindestens **7 Kanäle** (inkl. EEG/EOG/EMG zur echten Schlafphasenbestimmung), aber ohne Überwachungspersonal bei dem Patienten zu Hause.
* **Level III (Modifizierte Portable Sleep Apnea Testing / HSAT / Portable Monitor [PM]):** Mindestens **4 Kanäle** ohne EEG (typischerweise **2 Atmungskanäle [z. B. Airflow + RIP-Gurte oder Thorax- + Abdomen-Effort / EIP] + 1 Kardio-Kanal [HR oder EKG] + `SpO2`**). **Genau in diese Level-III-Kategorie fällt das Portable-Monitor-Setup aus Alarcón et al. (2023)!**
* **Level IV (Kontinuierliche 1- oder 2-Kanal-Aufzeichnung / Consumer Wearables):** Nur **1 bis 2 Parameter** (z. B. reines Finger-Pulsoximeter mit `SpO2` + Pulsrate, Smartwatch mit PPG + Accelerometer oder reine Schnarch-App).

#### 2. Klinische Screening-Fragebögen als „Mixed Multimodal“-Features
Wie in **Salvi et al. (2024)** gezeigt, lassen sich die Vorhersagen von Biosignal-Modellen signifikant verbessern, wenn man einfache klinische Basisdaten (*Tabular / Clinical Features*) mit einspeist. In der Schlafmedizin nutzt man dafür den **STOP-BANG-Score (0 bis 8 Punkte, ab `>= 3` Punkten erhöhtes OSA-Risiko)**:
* **S**noring: Schnarchen Sie laut? (Ja = 1)
* **T**ired: Sind Sie tagsüber oft müde/schläfrig? (Ja = 1)
* **O**bserved: Hat jemand beobachtet, dass Sie im Schlaf aufhören zu atmen? (Ja = 1)
* **P**ressure: Haben Sie Bluthochdruck (arterielle Hypertonie)? (Ja = 1)
* **B**MI: Ist Ihr Body-Mass-Index **`> 35 kg/m²`**? (Ja = 1)
* **A**ge: Sind Sie **älter als `50 Jahre`**? (Ja = 1)
* **N**eck: Ist Ihr Halsumfang **`> 40 cm`**? (Ja = 1)
* **G**ender: Ist Ihr biologisches Geschlecht **männlich**? (Ja = 1)
* *(Hinweis: Die vier objektiven Variablen **Alter, Geschlecht, BMI und Bluthochdruck/Schnarchen** sind in fast allen öffentlichen Datenbanken wie SHHS, MESA, UCDDB und ISRUC als Metadaten pro Patient vorhanden und können in eurer lokalen GUI als optionale Eingabefelder ergänzt werden!)*

#### 3. Die richtigen Evaluations-Metriken für euer Paper (Methodik für WP 5)
In medizinischen Klassifikationsproblemen mit starker **Klassen-Unwucht** (z. B. 85 % normale Atmungsfenster vs. 10 % OSA vs. 4 % Hypopnoe vs. 1 % CSA; oder im Schlafstadien-Scoring nur 5 % Stadium N1) ist die bloße **Accuracy** irreführend. Verwendet in eurem Paper daher dieses vollständige Metrik-Set:

* **Für die Fenster-/Event-Klassifikation (Binär & Multiklassen):**
  1. **Sensitivity (Recall / True Positive Rate):** `Sens = TP / (TP + FN)` -> *Wie viele echte Apnoe-Events findet das Modell? (Klinisches PM-Ziel laut Collop et al., 2011: `>= 82,5 %`)*
  2. **Specificity (True Negative Rate):** `Spec = TN / (TN + FP)` -> *Wie sicher erkennt das Modell normale Atmung ohne Fehlalarme?*
  3. **Precision (Positive Predictive Value, PPV):** `PPV = TP / (TP + FP)`
  4. **Negative Predictive Value (NPV):** `NPV = TN / (TN + FN)` -> *Wie sicher ist ein unauffälliger Befund wirklich gesund? (Für Screening-Tools extrem wichtig!)*
  5. **Positive Likelihood Ratio (`LR+`):** `LR+ = Sensitivity / (1 - Specificity)` -> *Klinisches PM-Ziel laut Alarcón et al. (2023): `LR+ >= 5,0`.*
  6. **Macro-F1-Score:** Ungewichteter Mittelwert der F1-Scores (`2 * Prec * Sens / (Prec + Sens)`) über alle Klassen hinweg -> bestraft Modelle, die seltene Klassen (wie Hypopnoe, CSA oder Schlafstadium N1) ignorieren!
  7. **Cohen's Kappa (`κ`):**
     `κ = (p_o - p_e) / (1 - p_e)`
     *(mit `p_o` = beobachtete Übereinstimmung, `p_e` = zufällige erwartete Übereinstimmung)*. Standard-Metrik für Schlafstadien und AHI-Schweregrad-Klassen (`κ > 0,60` = substanziell, `κ > 0,80` = exzellent / Experten-Niveau; für geordnete AHI-Schweregrade `Normal < Mild < Moderat < Schwer` nutzt man **Linear / Quadratic Weighted Kappa** wie in Alarcón et al., 2023, Table 5!).
  8. **AUROC** (Fläche unter der ROC-Kurve) & **AUPRC** (Fläche unter der Precision-Recall-Kurve, besonders aussagekräftig bei unbalancierten Testdaten).

* **Für die Patienten-Level AHI-Schätzung (`AHI_predicted` vs. `AHI_ground_truth`):**
  1. **Bestimmtheitsmaß (`R²`) & Pearson-/Intraklassen-Korrelation (`r` / `ICC`):** Misst den linearen Zusammenhang zwischen vorhergesagtem und wahrem AHI.
  2. **MAE (Mean Absolute Error) & RMSE (Root Mean Squared Error):** Durchschnittlicher absoluter AHI-Fehler in `Events/Stunde`.
  3. **Bland-Altman-Analyse (Goldstandard in der Medizintechnik!):** Plottet die Differenz `(AHI_pred - AHI_true)` gegen den Mittelwert `(AHI_pred + AHI_true) / 2` und zeigt den **systematischen Bias (Mittelwert der Differenz)** sowie die **95 % Limits of Agreement (`Bias ± 1,96 * SD`)**. Damit könnt ihr im Paper visuell beweisen, dass eure `TST`-Korrektur den systematischen Überschätzungs-Bias aus *Alarcón et al. (2023)* beseitigt!
