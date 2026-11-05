# Tiefenrecherche & Architektur-Kompendium (Simple & Modular Edition – PDF-Druckversion): Multimodale KI-gestützte Schlafanalyse und Binäre Schlafapnoe-Detektion (Teamprojekt AIN7)

## Inhaltsverzeichnis
1. **Projektkontext, Prüfungs-Constraints & Der vereinfachte „Paper-Übungs“-Ansatz (HTWG Konstanz)**
2. **Detaillierte Einzelanalyse der 6 bereitgestellten Kern-Paper (2021–2025)**
3. **Medizinisches & Theoretisches Tiefen-Kompendium: Schlafphysiologie, 5 Schlafphasen, Schlafapnoe, Andere Schlafstörungen (ICSD-3) & Klinische Konzepte**
4. **Biophysik, Sensorik & Signalgewinnung in der Schlafmedizin (EEG, EOG, EMG, EKG/HRV/EDR, SpO2/PPG, Atemfluss, RIP/EIP, Akustik, Aktigraphie)**
5. **Öffentliche Datensätze, Schlanke Multi-Dataset-Pipeline & Variables Sensor-Toggle-System (Sensoren flexibel aktivieren/deaktivieren)**
6. **Simples Modell-Design (Binär: Apnoe vs. Keine Apnoe) & Automatisierte Auswertungs-Pipeline (Tabellen & Plots fürs Paper)**
7. **Vereinfachte Arbeitsverteilung für 6 Personen (Work Packages 1–6) & Entspannter Zeitplan bis Ende Dezember 2026**
8. **Zusatz & Ausblick (Optional / Future Work): Was wäre nötig für (A) Multi-Klassen Apnoe-Typen (OSA/CSA/Hypopnoe) und (B) Einbezug eines Schlafphasen-Modells?**

---

## 1. Projektkontext, Prüfungs-Constraints & Der vereinfachte „Paper-Übungs“-Ansatz

### 1.1 Offizielle Leitfrage & Angepasster Projekt-Scope (Reine Forschungs- & Paper-Übung ohne Endanwendung)
* **Leitfrage:** *„How can we leverage multimodal data to boost sleep disorder classification using machine learning?“*
* **Klärung des Anforderungsniveaus durch den Professor:**
  Das Teamprojekt muss **keine hochkomplexen offenen Forschungsprobleme lösen**, nicht zwingend die wissenschaftlichen Limitationen des Lehrstuhl-Papers (*Alarcón et al., 2023*) übertreffen und **am Ende muss auch keine Endanwendung oder GUI entstehen**. Stattdessen dient das Projekt als **reine Forschungs- und Schreib-Übung für ein wissenschaftliches Paper**: Ihr braucht am Ende lediglich eine saubere **Daten- und Experiment-Pipeline**, die euch belastbare **Ergebnisdaten, Auswertungstabellen (CSV / LaTeX / Excel) und Plots** liefert, auf die ihr euch beim Schreiben eures KES-Papers stützen könnt!
* **Euer neuer, pragmatischer Kern-Ansatz („Simple, Modular & Research-Only“):**
  1. **Starke, vereinheitlichte Multi-Dataset-Pipeline:** Ihr baut eine saubere Python-Pipeline, die multimodale Schlafdaten aus **verschiedenen öffentlichen Datensätzen** (z. B. *UCDDB*, *ISRUC-Sleep*, *Apnea-ECG*, *MIT-BIH*, *CinC 2018* oder *SHHS/MESA*) in ein **einheitliches, simples 60-Sekunden-Fenster-Format** überführt.
  2. **Rein binäre Klassifikation (`0 = Keine Apnoe` vs. `1 = Apnoe`):** Euer Modell muss pro Zeitfenster lediglich entscheiden, ob eine **Apnoe vorliegt (`1`) oder nicht (`0`)** – genau wie im Ablauf von *Alarcón et al. (2023)*, aber ohne unnötigen Ballast.
  3. **Variables Sensor-Toggle-System (Sensoren flexibel an- und ausschaltbar):** Über eine zentrale Konfiguration (`ACTIVE_CHANNELS`) könnt ihr während der Entwicklung und für eure Paper-Experimente **beliebige Sensordaten (z. B. `SpO2`, `HR`, `Thorax-Gurt`, `Abdomen-Gurt`, `Airflow`, `EEG`, `EKG`) flexibel aktivieren oder deaktivieren**.
  4. **Automatisierte Auswertungstabellen & Paper-Plots statt GUI:** Ein Auswertungsskript läuft automatisch über alle definierten Sensor-Kombinationen und Datensätze und speichert direkt die **fertigen Vergleichstabellen (`results_sensor_ablation.csv`, `results_cross_dataset.csv`)** sowie die **publikationsreifen Grafiken (ROC-Kurven, AHI-Scatterplot, 1D-Grad-CAM- & Feature-Importance-Plot)** für euer Paper!
* **Vorgegebene 5 Projektphasen laut Kick-off-Präsentation (`Team_project_AIN_26WS`):**
  1. *Basics of AI:* Grundprinzipien (Rule-Based Systems vs. Classic Machine Learning mit hand-designed Features vs. Deep Learning nach Goodfellow et al., 2016).
  2. *Basic Knowledge on Sleep, Sleep Apnea & Multimodality:* Physiologie des Schlafs, Polysomnographie (PSG), Schlafapnoe und multimodale Sensorik.
  3. *State-of-the-Art Review:* Systematischer Vergleich bestehender KI-Ansätze für Schlafapnoe und Schlafanalyse.
  4. *Development of ML Pipeline:* Implementierung der eigenen modularen Multi-Dataset-Pipeline und des binären Apnoe-Klassifikators in Python.
  5. *Analysis & Critical Evaluation:* Auswertung der verschiedenen Sensor-Kombinationen (Ablationsstudie), Vergleich mit der Literatur und Ausblick auf mögliche Erweiterungen (Apnoe-Subtypen & Schlafphasen-Kopplung, siehe Kapitel 9).

### 1.2 Forschungsprofil eurer Betreuer am Ubiquitous Computing Lab (UC Lab, HTWG Konstanz)
Auch beim vereinfachten Projektansatz hilft es enorm zu wissen, wie eure drei Betreuer arbeiten, weil ihr euer Paper methodisch genau an deren Stil anlehnen könnt:
* **Prof. Dr. Ralf Seepold** (`ralf.seepold@htwg-konstanz.de`, Leiter UC Lab & Interdisziplinäres Schlaflabor): Co-Autor von **Gaiduk et al. (2023)** und **Alarcón et al. (2023)**. Sein Paper *Alarcón et al. (2023)* ist die perfekte **Blaupause für den Aufbau eures Papers**: 60-Sekunden-Fenster, binäre Klassifikation (`Apnoe` vs. `Keine Apnoe`), klare Gegenüberstellung verschiedener Vorverarbeitungs- oder Sensor-Setups, Auswertung über mehr als einen Datensatz und anschauliche Visualisierung der Ergebnisse.
* **Dr. Giulia Maria Mattia** (`gmattia@htwg-konstanz.de`): Spezialistin für multimodales Machine Learning / Deep Learning und Modell-Nachvollziehbarkeit. Euer **flexibles Sensor-Toggle-System** (mit dem ihr systematisch untersucht, welche Sensorkombination wie viel Genauigkeit bringt) passt exakt zu ihrem Forschungsgebiet!
* **MSc Daniel Vélez Gutiérrez** (`jvelezg@htwg-konstanz.de`): Forscht am UC Lab an nicht-invasiver Biosignal-Akquisition, tragbaren Sensoren und Daten-Pipelines.

---

### 1.3 Prüfungsleistungen & Formale Vorgaben (aus `Team_project_Evaluation`)
Jedes der 6 Teammitglieder erhält eine **individuelle Note (Skala 1 bis 5)**, basierend auf:

| Teilprüfung | Gewichtung | Formale Anforderungen & Constraints | Kritische Konsequenz für euer 6er-Team |
| :--- | :--- | :--- | :--- |
| **Wissenschaftliches Paper** | **60 %** | • **Nur 1 gemeinsames Paper pro Gruppe**<br>• Format: **KES Conference Paper Template** (Elsevier Procedia Computer Science / Springer KES)<br>• Seitenumfang: **Minimum 8, Maximum 10 Seiten** (exkl. Literaturverzeichnis; max. ca. 7.000–8.000 Wörter)<br>• **Maximal 5 grafische Elemente insgesamt (Tabellen UND Abbildungen zusammen!)** | **Striktes 5-Elemente-Limit beachten:** Ihr dürft im gesamten Paper zusammen nur höchstens 5 Tabellen/Abbildungen verwenden! Mit eurem simplen Sensor-Toggle-Ansatz lassen sich diese 5 Grafiken extrem sauber und überzeugend füllen (siehe Abschnitt 7.3). |
| **Mündliche Präsentation** | **40 %** | • **1 gemeinsame Gruppenpräsentation**<br>• **15 Minuten Redezeit pro Teammitglied** (= **90 Minuten Gesamtvortrag** bei 6 Personen!) | Jede Person braucht ein klar abgegrenztes **eigenes Arbeitspaket (Work Package)** entlang des Paper-Ablaufs (Daten-Pipeline, Sensor-Kombinationen, Klassisches ML vs. 1D-CNN, Multi-Dataset-Evaluation, GUI & XAI). |

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
* **Zielsetzung:** Entwicklung eines erklärbaren Deep-Learning-Modells zur Detektion obstruktiver Schlafapnoe-Events (OSA) und zur Berechnung des **Apnoe-Hypopnoe-Index (AHI)** für den Einsatz auf einem tragbaren Monitor (PM). **Dieses Paper dient euch als direktes Vorbild für den strukturellen Ablauf eures eigenen Papers!**
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
  * **Forderung nach Uncertainty Quantification (Konfidenzschätzung):** Für den praktischen klinischen Einsatz muss die Software anzeigen, bei welchen Epochen die Klassifikation **unsicher (mehrdeutig)** ist, damit der Arzt im GUI gezielt nur diese Stellen nachprüfen muss.

---

#### Paper 5: Salvi, Loh, Seoni, Barua, García, Molinari & Acharya (2024)
* **Titel:** *Multi-modality approaches for medical support systems: A systematic review of the last decade*
* **Journal & Link:** *Information Fusion*, Vol. 103, 102134, Elsevier (2024) — `https://doi.org/10.1016/j.inffus.2023.102134`
* **Fokus:** Systematisches PRISMA-Review über **81 Q1-Journal-Studien (2012–2022)** zu multimodaler Datenfusion in der Medizin.
* **Kernaussagen & Fusions-Taxonomie:**
  * **3 Datentyp-Kategorien:** (1) *Bioimaging* (MRT, CT, Röntgen), (2) *Biosignals* (EEG, EKG, EOG, EMG, PPG, SpO2), (3) *Mixed* (Kombination aus Biosignalen/Bildern mit **klinischen/demografischen Tabellendaten** wie Alter, Geschlecht, BMI, Fragebögen).
  * **Dokumentierte Forschungslücke im Bereich Schlaf (S. 10):** Die Autoren halten explizit fest: *„Finally, for sleep health, only biosignals-based approaches are utilized, with no consideration for clinical or demographic patient data.“* -> Selbst in einem simplen Modell könnt ihr optional `Alter`, `Geschlecht` und `BMI` als zusätzliche Spalten in eurer Feature-Tabelle aktivierbar machen!
  * **Zwei Haupt-Fusionsstrategien:**
    1. **Feature-Level Fusion (Early / Intermediate Fusion):** Zusammenführen der Kanäle oder ihrer extrahierten Merkmale (z. B. durch einfaches Aneinanderhängen der Kanal-Matrizen `[N_windows, N_active_channels, 60]` oder der statistischen Kanal-Features) vor dem Klassifikator. **Das ist der einfachste und direkteste Weg für euer Projekt!**
    2. **Decision-Level Fusion (Late Fusion):** Jede Modalität besitzt einen unabhängigen Klassifikator; die Wahrscheinlichkeits-Outputs werden über Weighted Averaging, Voting oder Stacking fusioniert. Vorteil: Extrem flexibel, wenn einzelne Sensoren fehlen (*Missing Modalities*).

---

#### Paper 6: Secinaro, Calandra, Secinaro, Muthurangu & Biancone (2021)
* **Titel:** *The role of artificial intelligence in healthcare: a structured literature review*
* **Journal & Link:** *BMC Medical Informatics and Decision Making*, Vol. 21, 125 (2021) — `https://doi.org/10.1186/s12911-021-01488-9`
* **Fokus:** Strukturierte Literaturübersicht und bibliometrische Analyse (288 Studien) zu KI im Gesundheitswesen.
* **Kernaussagen für euer Projekt:**
  * Identifiziert 5 Hauptcluster: *Health Services Management, Predictive Medicine, Patient Data, Diagnostics, Clinical Decision-Making*.
  * Diskutiert die Gefahr des **„De-skilling“ von Ärzten** durch blinde Black-Box-Automatisierung sowie die kritische Bedeutung von **Data Quality Management**, **Transparenz / Explainability** und **Datenschutz (Data Privacy / Governance)**. Dies liefert die direkte wissenschaftliche Begründung für den Einsatz von **Explainable AI (Grad-CAM & Feature Importance)** in unserer Auswertungs-Pipeline.

---

## 3. Medizinisches & Theoretisches Tiefen-Kompendium: Schlafphysiologie, Schlafphasen, Schlafapnoe, Andere Schlafstörungen (ICSD-3) & Klinische Konzepte

Dieses Kapitel liefert euch das vollständige medizinische, physiologische und methodische Fundament für **Phase 1 & 2 eures Projekts** (*„Acquire basic knowledge on sleep, sleep disorders, sleep apnea detection, multimodality and AI evaluation“*). Alle Themenfelder sind hier in voller Tiefe erhalten, damit ihr für das Schreiben von *Introduction*, *Background* und *Discussion* eures Papers sowie für Fragen in der 90-minütigen Prüfung jedes Detail sofort griffbereit habt.

---

### 3.1 Physiologie des Schlafs, Zwei-Prozess-Modell & Das Hypnogramm

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

#### C. Alle klinischen Schlafarchitektur-Kennzahlen
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

### 3.2 Die 5 Schlafphasen im Detail: R&K (1968) vs. AASM, Physiologie & Übergangs-Logik

#### A. Historische Entwicklung: Rechtschaffen & Kales (1968) vs. AASM-Standard (ab 2007 bis heute)
Da ältere öffentliche Datenbanken (wie *UCDDB*, *Sleep-EDF*, *MIT-BIH*, *CAP*) noch nach dem **R&K-Standard (1968)** annotiert wurden, neuere (wie *CinC 2018*, *ISRUC*, *SHHS-AASM*) jedoch nach **AASM**, ist diese Gegenüberstellung essenziell:

| Merkmal | Rechtschaffen & Kales (R&K, 1968) | AASM Manual (Version 2.6 / 3.0) | Umrechnungsregel für die Daten-Pipeline |
| :--- | :--- | :--- | :--- |
| **Anzahl der Stadien** | **6 Schlafstadien + Movement Time (MT)**:<br>`W`, `S1`, `S2`, `S3`, `S4`, `REM`, `MT` | **5 Stadien**:<br>`W` (Wake), `N1`, `N2`, `N3`, `R` (REM) | • `S1 -> N1`<br>• `S2 -> N2`<br>• **`S3 + S4 -> N3`** (Zusammenlegung!)<br>• `REM -> R`<br>• `MT` -> verwerfen oder der vorangehenden/Wach-Klasse zuordnen |
| **Tiefschlaf-Grenze** | `S3`: `20 % – 50 %` Delta-Wellen pro Epoche<br>`S4`: `> 50 %` Delta-Wellen pro Epoche | **`N3`**: Sobald **`>= 20 %`** einer 30s-Epoche (`>= 6 s`) Delta-Wellen (`0,5–2 Hz`, `> 75 µV` frontal) zeigen | Keine Trennung zwischen S3 und S4 mehr nötig, da physiologisch identisch. |
| **Empfohlene EEG-Kanäle** | Nur **1 zentraler Kanal** Pflicht (`C3-A2` oder `C4-A1`) | **3 Kanäle aus 3 Regionen** Pflicht:<br>Frontal (`F4-M1`), Zentral (`C4-M1`), Okzipital (`O2-M1`) | Falls ein Datensatz alle 3 Regionen hat: nutze `F4` (Delta/K-Komplex), `C4` (Spindeln/Apnoe) und `O2` (Alpha/Arousal); falls nur 1 Kanal gewählt wird: **`C4-M1` (`C4-A1`)**. |

#### B. Ganzkörper-Physiologie & AASM-30s-Scoring-Regeln über die 5 Schlafstadien hinweg

| Physiologisches System | Wach (`W`) | Leichtschlaf (`N1` ~5 % & `N2` ~45–55 %) | Tiefschlaf (`N3` / Slow-Wave Sleep, ~15–25 %) | Traumschlaf (`R` / REM-Schlaf, ~20–25 %) |
| :--- | :--- | :--- | :--- | :--- |
| **AASM-30s-Epochen-Regel & Gehirn (EEG)** | **`> 50 %` der Epoche** okzipitaler Alpha-Rhythmus (`8–12 Hz` bei geschlossenen Augen) oder Beta (`13–30 Hz`) | **N1:** Alpha `< 50 %`, ersetzt durch **LAMF (`4–7 Hz` Theta)** & Vertex-Wellen (`Cz`)<br>**N2:** Mind. 1 **K-Komplex (`>= 0,5 s` frontal)** und/oder **Schlafspindel (`11–16 Hz` zentral)**; `< 20 %` Delta | **Mind. `>= 20 %` der Epoche (`>= 6 s`)** hochamplitudige **Delta-Wellen (`0,5–2 Hz`, `> 75 µV` frontal)**, höchste kortikale Synchronisation | Desynchronisiertes Theta/Beta + **Sägezahnwellen (`2–6 Hz`)**; Gehirnstoffwechsel fast so hoch wie im Wachzustand! |
| **Augen (EOG)** | Schnelle Blicksprünge & Blinzeln (`0,5–2 Hz`) | **N1:** Langsame rollende Augenbewegungen (**SEMs**, `0,2–0,5 Hz`)<br>**N2:** Keine Augenbewegungen | Keine Augenbewegungen (EOG zeigt nur eingestreute Stirn-Delta-Wellen) | **Phasische Bursts schneller, sakkadischer Augenbewegungen (REMs)** |
| **Muskeltonus (Kinn-EMG)** | Hoch und variabel | Moderat (niedriger als im Wachzustand) | Niedrig (weiter abgesenkt) | **Komplette Skelettmuskel-Atonie (Muskellähmung bis auf Zwerchfell & Augenmuskeln!)** |
| **Herz & Kreislauf (EKG / Blutdruck)** | Hoch, variabel (Sympathikus-dominant) | Absinken von Herzfrequenz und Blutdruck; **Parasympathikus (Vagus)** übernimmt | **Tiefster Punkt von Herzfrequenz und Blutdruck („Nächtliches Dipping“ um 10–20 %)**, extrem regelmäßige R-R-Intervalle (hohe HF-HRV) | **Vegetativer Sturm (Instabilität):** Herzfrequenz und Blutdruck schwanken stark zwischen Bradykardie und Tachykardie (Sympathikus-Bursts) |
| **Atmung & Obere Atemwege** | Willkürlich + metabolisch gesteuert; hoher Tonus der Rachenmuskulatur | Rein metabolisch (CO2-gesteuert); im Übergang W->N1 oft leicht periodisch; Rachentonus sinkt leicht | **Stabilste, gleichmäßigste Atmung der ganzen Nacht!** Zentrale Chemoreflex-Steuerung ist ruhig -> **wenigste Apnoen in N3!** | Unregelmäßige Atemfrequenz & Atemzugtiefe; **Tonus des Zungenmuskels (*M. genioglossus*) fällt auf nahezu Null** -> **längste und schwerste obstruktive Apnoen in REM!** |

#### C. Biologische Übergangs-Grammatik zwischen Schlafphasen
Schlafstadien wechseln nicht völlig zufällig von einer 30s-Epoche zur nächsten, sondern folgen biologischen Regeln:
* **Hohe Selbst-Übergangswahrscheinlichkeit:** Wenn Epoche `t` in `N2`, `N3` oder `REM` ist, liegt die Wahrscheinlichkeit, dass Epoche `t+1` im selben Stadium bleibt, bei `> 85–92 %`.
* **Biologisch unplausible Sprünge:** Ein gesunder Mensch springt niemals direkt von `Wake (W)` in `Tiefschlaf (N3)` oder direkt von `Tiefschlaf (N3)` in `REM` (dazwischen liegt fast immer `N1`/`N2`). Umgekehrt kann man aus **jedem** Schlafstadium (`N1, N2, N3, REM`) durch ein Arousal sofort nach **`Wake (W)`** oder **`N1`** springen!

---

### 3.3 Schlafapnoe & Schlafbezogene Atmungsstörungen (SBAS) – Pathophysiologie, Subtypen & Indizes

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

#### C. Alle klinischen Schweregrad-Indizes & Biomarker

1. **AHI (Apnea-Hypopnea Index – Goldstandard mit bekannter Schlafzeit `TST`):**
   `AHI [Events/h] = (Anzahl aller Apnoen [OSA + CSA + MSA] + Anzahl aller Hypopnoen) / Total Sleep Time (TST in Stunden)`
2. **REI (Respiratory Event Index – Standard für Portable Monitors ohne Schlafzeit-Erkennung, wie in unserem simplen Basis-Modell):**
   `REI [Events/h] = (Anzahl aller erkannten Apnoe-Fenster) / Total Recording Time (TRT in Stunden)`
   *(Hinweis: Im Paper von Alarcón et al., 2023 wird genau dieser Quotient aus erkannten 60s-Apnoe-Fenstern geteilt durch 8 Stunden Aufnahmedauer als `AHI_predicted` bezeichnet!)*
3. **RDI (Respiratory Disturbance Index – inkl. RERAs):**
   `RDI [Events/h] = (Anzahl Apnoen + Anzahl Hypopnoen + Anzahl RERAs) / TST [Stunden]`
4. **Klinische Schweregrad-Klassifikation bei Erwachsenen (nach AASM):**
   * **Normal / Physiologisch:** `AHI < 5,0 Events/h`
   * **Leichte (Milde) Schlafapnoe:** `5,0 <= AHI < 15,0 Events/h`
   * **Mittelschwere (Moderate) Schlafapnoe:** `15,0 <= AHI < 30,0 Events/h`
   * **Schwere (Severe) Schlafapnoe:** `AHI >= 30,0 Events/h`
5. **Ergänzende Oximetrie- & Phänotyp-Metriken:**
   * **ODI3% / ODI4% (Oxygen Desaturation Index):** Anzahl der SpO2-Abfälle um `>= 3 %` bzw. `>= 4 %` pro Stunde.
   * **T90 (Time below 90 % SpO2):** Prozentualer Anteil der Zeit, in der die Sauerstoffsättigung unter `< 90 %` lag.
   * **MAD (Mean Apnea-Hypopnea Duration):** Durchschnittliche Dauer eines respiratorischen Events in Sekunden.
   * **Hypoxic Burden (`% * min / h`):** Fläche zwischen der SpO2-Baseline und der SpO2-Kurve während aller Events, geteilt durch die Schlafzeit in Stunden.
   * **REM-AHI vs. NREM-AHI:** Separate Berechnung des AHI im REM-Schlaf (`AHI_REM`) und im NREM-Schlaf (`AHI_NREM`).
   * **Supine-AHI vs. Non-Supine-AHI:** Verhältnis der Apnoen in Rückenlage vs. Seitenlage (*Positional OSA*).

---

### 3.4 Weitere relevante Schlafstörungen nach ICSD-3

Hier ist die vollständige Übersicht der **6 Hauptkategorien nach ICSD-3 (*International Classification of Sleep Disorders*)** und ihrer typischen Signal-Signaturen in der PSG:

| ICSD-3 Krankheitsgruppe | Wichtigste Krankheitsbilder | Primäre diagnostische Sensoren in der PSG | Charakteristische Biomarker & Signal-Signatur | Öffentliche Datensätze mit diesen Labels |
| :--- | :--- | :--- | :--- | :--- |
| **1. Insomnien** | Chronische Ein- und Durchschlafstörung (*Insomnia Disorder*) | EEG, EOG, Kinn-EMG, Aktigraphie | • **Makroarchitektur:** Einschlaflatenz `SOL > 30 Min.`, Wachzeit nach Einschlafen `WASO > 30–45 Min.`, Schlafeffizienz `SE < 85 %`, verkürzte `TST < 6 h`, reduzierte N3-Dauer.<br>• **Mikroarchitektur (EEG-Hyperarousal):** Abnorm erhöhte relative **Beta- (`15–30 Hz`) und Gamma-Power (`30–45 Hz`) während des NREM-Schlafs** sowie erhöhte Herzfrequenz und reduzierte HF-HRV in der Nacht. | **CAP Sleep Database** (`ins` Subgruppe), **Sleep-EDF** (Subgruppe `ST` mit Einschlafstörungen), **ISRUC-Sleep** |
| **2. Schlafbezogene Atmungsstörungen (SBAS / SDB)** | **OSA**, **CSA**, **MSA**, Cheyne-Stokes-Atmung, Schlafbezogene Hypoventilation | Thermistor, Nasaldruck, RIP-Gurte (`Thor/Abdo`), `SpO2`, EKG, EEG, Mikrofon | • `>= 90 %` (Apnoe) bzw. `>= 30 %` (Hypopnoe) Atemfluss-Reduktion für `>= 10 s`<br>• Paradoxe RIP-Gurtbewegung (OSA) vs. flache Gurte (CSA)<br>• Zyklische SpO2-Entsättigungen (`ODI >= 5/h`)<br>• CVHR im EKG (VLF-Peak bei `0,01–0,03 Hz`) + EEG-Arousals (`> 3 s` Alpha/Beta). | **UCDDB**, **CinC 2018**, **SHHS1/2**, **MESA**, **ISRUC**, **PSG-Audio**, **Apnea-ECG**, **MIT-BIH** |
| **3. Zentrale Störungen mit Tagesschläfrigkeit (Hypersomnien)** | **Narkolepsie Typ 1** (mit Kataplexie / Orexin-Mangel) & **Typ 2**, Idiopathische Hypersomnie | EEG, EOG, Kinn-EMG (Nacht-PSG + Tagestest MSLT) | • **Extrem kurze Einschlaflatenz (`SOL < 8 Min.`)**<br>• **SOREMP (*Sleep-Onset REM Period*):** Übergang von Wach/N1 direkt in den **REM-Schlaf innerhalb von `<= 15 Minuten`** nach dem Einschlafen!<br>• Stark fragmentierter Nachtschlaf mit häufigen kurzen Wach-Episoden. | **CAP Sleep Database** (`narco` Subgruppe), **MNC (Stanford/NSRR Narcolepsy Cohort)** |
| **4. Schlafbezogene Bewegungsstörungen** | **PLMD (*Periodic Limb Movement Disorder*)**, **RLS (*Restless Legs Syndrome*)**, **Schlaf-Bruxismus** (Zähneknirschen) | **Tibialis-EMG** (Beine), **Kinn-/Masseter-EMG** (Kiefer), EEG, EKG/HR | • **PLMS (Beinbewegungen):** Repetitive Bursts im Schienbein-EMG von `0,5–10 s` Dauer (Anstieg `>= 8 µV` über Ruhe-EMG), mindestens `4` Bursts in Folge im typischen Intervall von **`5 bis 90 Sekunden`** (oft alle `20–40 s` in N1/N2), begleitet von kurzen Herzfrequenz-Anstiegen und EEG-Arousals. Pathologisch ab **`PLMI > 15 Events/h`**.<br>• **Bruxismus:** Phasisches (`>= 3 Bursts` à `0,25–2 s`) oder tonisches (`> 2 s`) hochamplitudiges Knirschen im Kinn-/Masseter-EMG + Audio. | **CAP Sleep Database** (`plm` & `brux` Subgruppen), **ISRUC-Sleep**, **SHHS / MrOS** |
| **5. Parasomnien** | • **NREM-Parasomnien:** Schlafwandeln, Pavor nocturnus<br>• **REM-Parasomnie:** **RBD (*REM Sleep Behavior Disorder*)** | EEG (`F4, C4, O2`), EOG, **Kinn-EMG & Tibialis-EMG**, Video | • **NREM-Parasomnie:** Abruptes Erwachen direkt aus dem **Tiefschlaf (N3)** im ersten Nachtdrittel; das EEG zeigt weiterhin **hochamplitudige langsame Delta-Wellen (*Hypersynchronous Delta*)** gemischt mit Alpha/Beta.<br>• **RBD (REM-Schlaf-Verhaltensstörung):** **Verlust der physiologischen Muskelatonie im REM-Schlaf (*RSWA – REM Sleep Without Atonia*)**! Während EOG und EEG eindeutigen REM-Schlaf zeigen, weist das Kinn-/Bein-EMG hohe Muskelaktivität auf (wichtigster Frühindikator für **Parkinson**!). | **CAP Sleep Database** (`rbd` [22 Patienten!] & `nfle` [Nächtliche Frontallappen-Epilepsie, 40 Patienten!]) |
| **6. Zirkadiane Schlaf-Wach-Rhythmusstörungen** | Verzögerter (*DSWPD*) oder Vorverlagerter (*ASWPD*) Schlafphasentyp, Schichtarbeitersyndrom | Mehr-Tages-Aktigraphie, Hauttemperatur, Lichtsensor, PPG/HRV | • Phasenverschiebung des Aktivitäts-Ruhe-Rhythmus (*Cosinor-Analyse: Mesor, Amplitude, Acrophase*) und des nächtlichen Herzfrequenz-Minimums um `> 2–4 Stunden` gegenüber der sozialen Normzeit. | **MESA Actigraphy** (7 Tage Handgelenks-Aktigraphie parallel zur PSG!), **STAGES** |

---

### 3.5 Weitere relevante klinische & methodische Konzepte für das KES-Paper

#### 1. Die 4 klinischen Geräteklassen der Schlafdiagnostik (AASM Level I bis Level IV)
Mit eurem **Sensor-Toggle-System** könnt ihr im Paper direkt zeigen, dass ihr alle 4 AASM-Geräteklassen simulieren könnt:
* **Level I (Überwachte Labor-Polysomnographie / Standard-PSG):** Mindestens **7 Kanäle** (EEG, EOG, Kinn-EMG, EKG, Airflow, Atmungs-Effort, `SpO2`) im Schlaflabor -> *entspricht eurem Preset `"4_full_multimodal"`*.
* **Level II (Unüberwachte Voll-Polysomnographie zu Hause):** Mindestens **7 Kanäle** (inkl. EEG) zu Hause.
* **Level III (Portable Monitor [PM] / Home Sleep Apnea Testing [HSAT]):** Mindestens **4 kardiorespiratorische Kanäle** ohne EEG (z. B. `SpO2`, `HR`, `Thor-Res`, `Abdo-Res` oder `Airflow`) -> *entspricht exakt eurem Preset `"2_portable_monitor"` bzw. `"3_respiratory_plus"` (wie in Alarcón et al., 2023)*.
* **Level IV (1–2-Kanal-Screening / Consumer Wearable):** Nur **1 bis 2 Parameter** (z. B. `SpO2` + `HR` am Finger oder Handgelenk) -> *entspricht eurem Preset `"1_minimal_oximeter"`*.

#### 2. Klinische Screening-Fragebögen (STOP-BANG-Score)
Der **STOP-BANG-Score (0 bis 8 Punkte, ab `>= 3` Punkten erhöhtes OSA-Risiko)** umfasst:
* **S**noring (Lautes Schnarchen), **T**ired (Tagesschläfrigkeit), **O**bserved (Beobachtete Atemaussetzer), **P**ressure (Bluthochdruck), **B**MI (`> 35 kg/m²`), **A**ge (`> 50 Jahre`), **N**eck (Halsumfang `> 40 cm`), **G**ender (männlich).

#### 3. Klinische Benchmark-Ziele für Portable Monitors (nach Collop et al., 2011 & Alarcón et al., 2023)
Damit ein automatisierter Apnoe-Detektor auf einem Portablen Monitor (Level III) klinisch als valide gilt, muss er laut AASM-Richtlinien (*Collop et al., 2011*) mindestens folgende Schwellenwerte erreichen (die Formeln und der Python-Code aller Metriken folgen im Auswertungs-Kapitel bei `evaluate_predictions`):
* **Sensitivity (Trefferquote):** Mindestens **`>= 82,5 %`**
* **Positive Likelihood Ratio (`LR+ = Sensitivity / (1 - Specificity)`):** Mindestens **`LR+ >= 5,0`** (entspricht bei `82,5 %` Sensitivity einer **Specificity von `>= 83,5 %`**).

---

## 4. Biophysik, Sensorik & Signalgewinnung in der Schlafmedizin (Deep Dive)

Um multimodale Schlafdaten korrekt vorzuverarbeiten, sinnvolle Features zu extrahieren und die Modell-Entscheidungen medizinisch fundiert zu erklären, folgt hier die detaillierte Aufschlüsselung **aller Schlafsensoren**, ihrer **physikalischen Messprinzipien**, ihrer **Abtast- und Signalfrequenzen (in Hz)** sowie ihrer **spezifischen Muster bei Schlafstadien und Schlafapnoe**.

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
| **3. RIP-Gurte (`Thor-Res` & `Abdo-Res`)** | Zwei elastische Gurte um **Brustkorb (Thorax)** und **Bauch (Abdomen)** mit sinusförmig eingewebter Drahtspule; misst die **Änderung der magnetischen Selbstinduktivität `ΔL`** proportional zur Querschnittsfläche. | `fs = 10–100 Hz`<br>Filter: `0,1–15 Hz` | **Primärer Sensor für die Atemmechanik & Phasenlage:** Liefert das Amplituden- und Phasenverhältnis zwischen Brust- und Bauchatmung zur Trennung von obstruktiven vs. zentralen Events (exakte Kriterien siehe Tabelle in Abschnitt `Exakte Abgrenzung aller 5 respiratorischen Event-Typen`). Typische Artefakte: Verrutschen des Gurts bei Lagewechsel. |
| **4. Elektrische Impedanzpneumographie (EIP)** | Misst über normale EKG-Elektroden mittels hochfrequentem Schwachstrom (`20–100 kHz`) die **elektrische Impedanzänderung des Brustkorbs** durch das Luftvolumen in der Lunge. | `fs = 10–100 Hz` | Komfortabler Ersatz für RIP-Gurte in **Portablen Monitoren (PMs)** (wird u. a. im Ziel-PM des HTWG UC Lab verwendet, siehe Alarcón et al., 2023!). |

---

### 4.7 Akustik (Schnarch-/Trachealmikrofon) & Körperlage / Aktigraphie
* **Tracheal- und Raummikrofon (`fs = 8.000 – 48.000 Hz`):**
  * Erfasst Gewebevibrationen der verengten oberen Atemwege (**Schnarchen**, Hauptenergie zwischen `100 Hz` und `1.000 Hz`, wichtige Features: **MFCCs**, **800-Hz-Power-Ratio**, **Spectral Entropy**).
  * Charakteristisches OSA-Muster: Lautes rhythmisches Schnarchen -> **absolute akustische Stille während der Apnoe (`10–60 s`)** -> **explosives Post-Apnoe-Schnarchen / Luftschnappen (PAS)** beim Arousal.
* **Körperlagesensor & 3-Achsen-Accelerometer (`fs = 1 – 100 Hz`):**
  * Bestimmt über den Erdgravitationsvektor die Schlaflage (**Rückenlage [Supine]**, Seitenlage links/rechts [Lateral], Bauchlage [Prone]).
  * Da bei über 50 % der Patienten der Zungengrund vor allem in **Rückenlage** nach hinten fällt (**Positional OSA**), liefert die Korrelation von erkannten Apnoe-Events mit der Körperlage einen hohen diagnostischen Mehrwert als optionaler Zusatz-Kanal in der Pipeline!

---

## 5. Öffentliche Datensätze, Schlanke Multi-Dataset-Pipeline & Variables Sensor-Toggle-System

### 5.1 Übersicht der besten öffentlichen Datensätze für euer Projekt

#### Gruppe A: Sofort frei herunterladbar (Ohne Antrag / Wartezeit – ideal für den sofortigen Start!)
1. **UCDDB – St. Vincent's University Hospital / University College Dublin Sleep Apnea Database (PhysioNet):**
   * **Link:** `https://physionet.org/content/ucddb/1.0.0/`
   * **Umfang:** `25` vollständige Übernacht-PSGs (`6–8 h`) von Erwachsenen mit Verdacht auf Schlafapnoe (`AHI 1,7 bis 90`).
   * **Signale (`128 Hz`):** Volle PSG (14 Kanäle): EEG (`C3-A2`, `C4-A1`), EOG (2 ch), Kinn-EMG, EKG, Oronasaler Thermistor-Airflow, `Thor-Res` (RIP), `Abdo-Res` (RIP), `SpO2`, Pulsrate, Schnarchsensor, Körperposition.
   * **Annotationen:** Sekundengenaue Startzeit & Dauer für **OSA, CSA, MSA, Hypopnoe** sowie **30s-Schlafstadien** und **Demografie (Alter, BMI, Geschlecht, AHI)**!
2. **ISRUC-Sleep (University of Coimbra, Portugal):**
   * **Link:** `https://sleeptight.isr.uc.pt/`
   * **Umfang:** `118` Probanden (*Subgroup 1:* `100` Patienten mit Schlafstörungen/Apnoe; *Subgroup 2:* `8` Patienten mit je 2 Nächten; *Subgroup 3:* `10` Gesunde).
   * **Signale (`200 Hz`):** 6 EEG, 2 EOG, Kinn-EMG, Bein-EMG, EKG, Airflow, `Thor-Res`, `Abdo-Res`, `SpO2`, Schnarchmikrofon, Körperlage + klinische Patienten-Metadaten.
   * **Annotationen:** AASM-Schlafstadien von **2 unabhängigen Experten** + Apnoe-/Arousal-Events.
3. **PhysioNet / Computing in Cardiology Challenge 2018 („You Snooze You Win“ / Massachusetts General Hospital):**
   * **Link:** `https://physionet.org/content/challenge-2018/1.0.0/`
   * **Umfang:** **`1.983` Patienten** (`994` vollständig annotierte Trainings-PSGs + `989` Test-PSGs; einzelne Patientenordner können gezielt per Python-Skript heruntergeladen werden, z. B. 50–100 Patienten, man muss nicht die gesamten 140 GB laden!).
   * **Signale (alle einheitlich bei `200 Hz`!):** 6 EEG-Kanäle (`F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1`), EOG (`E1-M2`), Kinn-EMG, Abdomen, Chest (`Thor`), Airflow, `SaO2`, EKG.
   * **Annotationen:** **AASM-Schlafstadien (W, N1, N2, N3, R)** + exakte Event-Grenzen für **OSA, CSA, MSA, Hypopnoe** UND **RERA / Non-Apnea Arousals**!
4. **PhysioNet Apnea-ECG (`70` Aufnahmen, `8` davon [`a01er`–`a04er`, `b01er`, `c01er`–`c03er`] mit vollen Atmungssignalen + `SpO2` bei `100 Hz` und fertigen 60s-Apnoe-Labels `A` vs. `N`):** `https://physionet.org/content/apnea-ecg/1.0.0/`
5. **MIT-BIH Polysomnographic Database (`slpdb`, `16` Aufnahmen bei `250 Hz` mit 30s-Apnoe- und Schlafstadien-Labels):** `https://physionet.org/content/slpdb/1.0.0/`
6. **PSG-Audio (`212` komplette PSGs bei `200 Hz` inkl. Audio & Apnoe-Events):** `https://www.scidb.cn/en/detail?dataSetId=778740145531650048`
7. **Sleep-EDF Expanded (`197` PSGs bei `100 Hz`):** `https://physionet.org/content/sleep-edfx/1.0.0/`

#### Gruppe B: Große klinische Kohorten im NSRR (`sleepdata.org` – kostenloser Account + DUA-Antrag)
* **SHHS1 (`5.793` PSGs) & SHHS2 (`2.651` PSGs):** `https://sleepdata.org/datasets/shhs`
* **MESA (`2.060` PSGs):** `https://sleepdata.org/datasets/mesa`
* *(Praxis-Tipp für den simplen Ansatz: Ihr könnt **sofort ab Tag 1** mit **UCDDB (25 Patienten) + ISRUC (100 Patienten) + Apnea-ECG / CinC 2018 (z. B. 50 Patienten)** starten. Damit habt ihr bereits 3 verschiedene Datensätze ohne jede Wartezeit! Falls euer Betreuer euch zusätzlich die vorverarbeiteten SHHS2-/MESA-Dateien aus Alarcón et al. (2023) gibt, hängt ihr sie einfach als 4. und 5. Datensatz in denselben Adapter ein.)*

---

### 5.2 Die vereinfachte 4-Stufen-Datenpipeline für Multi-Dataset-Auswertung (Binär: Apnoe vs. Keine Apnoe)

Damit das Projekt technisch **simpel, leicht verständlich und extrem schnell umsetzbar** bleibt, aber trotzdem **viele modale Daten aus verschiedensten Datensätzen zusammen auswerten kann**, nutzt ihr ein **Adapter-Pattern**, das jeden Datensatz in exakt dasselbe **60-Sekunden-Fenster-Format (auf `1 Hz` wie bei Alarcón et al., 2023)** bringt:

```text
[Verschiedene Datensätze: UCDDB (.rec/.edf), ISRUC (.rec), CinC2018 (.mat), SHHS/MESA (.edf)]
                                             |
                                             v
  +-----------------------------------------------------------------------------------+
  | STUFE 1: Datensatz-Adapter & Kanonisches Kanal-Wörterbuch                         |
  | • Jeder Datensatz bekommt eine kleine Lade-Funktion (ca. 30 Zeilen Python/MNE),   |
  |   die vorhandene Kanäle auf 7 einheitliche Standard-Namen umbenennt:              |
  |   ["SPO2", "HR", "THOR", "ABDO", "FLOW", "EEG_C4", "ECG_AMP"]                     |
  | • Falls "HR" nicht als fertiger Kanal existiert, wird HR aus den EKG-R-Zacken     |
  |   pro Sekunde berechnet. Falls "EEG_C4" genutzt wird, wird pro Sekunde die        |
  |   mittlere Band-Power (z.B. Alpha/Beta-Arousal-Power 8-30 Hz) extrahiert!         |
  +-----------------------------------------------------------------------------------+
                                             |
                                             v
  +-----------------------------------------------------------------------------------+
  | STUFE 2: Einheitliches 1-Hz-Downsampling & Simpler Artefakt-Filter                |
  | • Wie in Alarcón et al. (2023) werden alle Kanäle einheitlich auf fs = 1 Hz       |
  |   (1 Wert pro Sekunde) gebracht! Dadurch hat jeder Kanal exakt dieselbe Länge!    |
  | • Artefakt-Bereinigung (lineare Interpolation):                                   |
  |   SpO2 < 70 % oder > 100 % -> interpolieren | HR < 40 oder > 200 bpm -> interpol. |
  +-----------------------------------------------------------------------------------+
                                             |
                                             v
  +-----------------------------------------------------------------------------------+
  | STUFE 3: Normalisierung & 60-Sekunden-Windowing                                   |
  | • Schneiden der Nacht in nicht-überlappende 60-Sekunden-Fenster                   |
  |   (480 Fenster bei 8 Stunden Schlaf; jedes Fenster hat 60 Werte pro Kanal).       |
  | • Normalisierung (wählbar per Flag):                                              |
  |   - Option A (Alarcón 2023): Fensterweiser Z-Score z = (x - mean) / (std + 1e-6)  |
  |   - Option B (Robuster für SpO2): Pro Patient (x - Median_Nacht) / IQR_Nacht      |
  +-----------------------------------------------------------------------------------+
                                             |
                                             v
  +-----------------------------------------------------------------------------------+
  | STUFE 4: Simples Binäres Labeling (0 = Keine Apnoe, 1 = Apnoe)                    |
  | • Alle respiratorischen Events eines Datensatzes (egal ob OSA, CSA, MSA oder      |
  |   Hypopnoe) werden zusammengefasst:                                               |
  |   - Liegen >= 10 Sekunden eines Events im 60s-Fenster -> Label y = 1 (Apnoe)      |
  |   - Sonst                                             -> Label y = 0 (Keine Apnoe)|
  | • Speichern als EINE einheitliche Datei pro Datensatz (.npz oder .parquet):       |
  |   X mit Shape (N_windows, 7_Kanäle, 60_Sekunden), y mit Shape (N_windows,)        |
  +-----------------------------------------------------------------------------------+
```

**Warum dieser 1-Hz-Trick (inkl. 1-Hz-EEG-Bandpower) das Projekt radikal vereinfacht:**
Normalerweise ist die Fusion von `SpO2` (`1 Hz`), `Atmung` (`10–32 Hz`) und `EEG` (`100–200 Hz`) kompliziert, weil jedes Signal unterschiedlich viele Datenpunkte pro Minute hat (`60` vs. `600` vs. `6.000` Punkte). Wenn ihr jedoch für langsame Signale (`SpO2`, `HR`, `THOR`, `ABDO`, `FLOW`) direkt auf `1 Hz` abtastet und für hochfrequente Signale (`EEG_C4`, `ECG`) pro Sekunde einfach die **sekündliche Energie/Amplitude** (`1 Wert pro Sekunde = 1 Hz`) speichert, hat **jeder einzelne Sensor exakt 60 Zahlen pro 60-Sekunden-Fenster**! Damit passen alle Sensoren aller Datensätze in einen einzigen, simplen 3D-NumPy-Array `X` der Form `(Anzahl_Fenster, Anzahl_Sensoren, 60)`!

---

### 5.3 Übersicht: Welcher Sensor hat welchen Einfluss & Qualitätsgewinn auf die Apnoe-Erkennung?

Bevor ihr im Code einzelne Sensoren an- und ausschaltet, findet ihr hier die kompakte Übersicht, **welcher Sensor welchen konkreten Einfluss auf die Erkennungsqualität hat**, **welche Fehler er verhindert** und **was passiert, wenn man ihn deaktiviert**. Diese Übersicht dient euch gleichzeitig als **direkte wissenschaftliche Begründung für eure Hypothesen und Diskussion im KES-Paper**:

| Sensor-Kanal (Pipeline-Name) | Was misst der Sensor bei einer Apnoe? | Konkreter Qualitätsgewinn beim AKTIVIEREN | Schwäche / Risiko beim DEAKTIVIEREN (oder wenn nur dieser Sensor allein aktiv ist) | Erwarteter Effekt laut Literatur (Referenzwerte fürs Paper) |
| :--- | :--- | :--- | :--- | :--- |
| **1. `SPO2`**<br>*(Sauerstoffsättigung am Finger)* | Erfasst die **Folge** des Atemstillstands: den **Sauerstoffabfall im Blut (`>= 3 %` bzw. `>= 4 %` Desaturierung)**. | **Stärkster Einzel-Prädiktor für hohe Spezifität & AHI-Schweregrad:** Fast jede klinisch relevante Apnoe/Hypopnoe ab mittlerem Schweregrad erzeugt einen charakteristischen V-förmigen `SpO2`-Trichter. Verhindert zuverlässig Fehlalarme bei harmlosen Atemschwankungen ohne Sauerstoffmangel. | • **Wird `SPO2` deaktiviert:** Die Spezifität bricht stark ein (viele False Positives bei harmlosen Bewegungen), und **Hypopnoen** (die laut AASM einen `>= 3 %` Drop brauchen) sind kaum noch sicher erkennbar.<br>• **`SPO2` allein ohne Atmungssensor:** Wegen der **`15–45 s` Kreislauf-Totzeit** (Lunge -> Finger) wird die Apnoe im falschen (nachfolgenden) 60s-Fenster erkannt; kurze Apnoen bei jungen/schlanken Patienten ohne tiefen O2-Drop werden übersehen (**niedrigere Sensitivity**). | • `SpO2` allein: ~`80–90 %` Acc, aber oft zeitverzögert (*Paul et al., 2022; Gutiérrez-Tobal et al., 2021*).<br>• In Grad-CAM (*Alarcón et al., 2023*) der **wichtigste Einzelkanal (~35–45 % Gewicht)**. |
| **2. `HR`**<br>*(Herzfrequenz aus EKG oder PPG)* | Erfasst die **autonome Stressreaktion (CVHR)**: **Bradykardie** (Puls fällt während des Atemstopps) gefolgt von einem steilen **Tachykardie-Peak** (Puls schießt beim Luftschnappen hoch). | **Beseitigt Finger-Artefakte & verbessert die zeitliche Schärfe:** Wenn `SPO2` durch einen kalten oder verrutschten Fingersensor kurz schwankt, zeigt `HR` keine Bradykardie-Tachykardie-Welle -> das Modell erkennt den `SPO2`-Drop sofort als Artefakt! Zudem reagiert `HR` **ohne 30s-Kreislaufverzögerung** exakt in der Sekunde des Arousals. | • **Wird `HR` deaktiviert:** Mehr Fehlalarme durch Bewegungs-Artefakte am Pulsoximeter; das Ende einer Apnoe (Arousal-Zeitpunkt) wird ungenauer erkannt.<br>• **`HR` allein:** Sehr unspezifisch (`~75–80 %`), da die Herzfrequenz auch durch Träume (REM), Lagewechsel oder Beinbewegungen (PLMS) ansteigt. | • `HR/EKG` allein: `80,0 %` Acc.<br>• Kombination `SPO2 + HR`: Steigert die Accuracy sofort auf **`91,8 %`** (*Paul et al., 2022*)! |
| **3. `THOR` & `ABDO`**<br>*(Brust- & Bauchgurt / RIP-Effort)* | Erfassen die **mechanische Atembewegung** von Brustkorb und Bauch (synchron im Normalzustand, **gegenphasig/paradox** bei OSA, **komplett flach** bei CSA). | **Massiver Gewinn an Sensitivity (Trefferquote) & exaktem Timing im 60s-Fenster:** Zeigt den Atemstillstand bzw. den frustranen Atemkampf **sofort in Sekunde 1** (lange bevor `SPO2` fällt!). Zusammen mit `SPO2 + HR` bildet dies das **exakte 4-Kanal-Portable-Monitor-Setup aus Alarcón et al. (2023)**. | • **Werden `THOR` & `ABDO` deaktiviert:** Das Modell sieht keine mechanische Atmung mehr (nur noch indirekte Blut-/Pulsfolgen) -> Sensitivity für kürzere Apnoen (`10–15 s`) sinkt deutlich; eine spätere Trennung von **OSA vs. CSA** (Kapitel 8.1) wird unmöglich. | • `SPO2 + HR + THOR + ABDO`: Erreicht auf SHHS2 **84,3 % Acc / 92,1 % AUC** (*Alarcón et al., 2023*). |
| **4. `FLOW`**<br>*(Nasen-/Mund-Atemfluss: Thermistor / Nasaldruck)* | Misst direkt den **tatsächlichen Luftstrom an Nase und Mund** (`>= 90 %` Abfall = Apnoe, `>= 30 %` Abfall = Hypopnoe). | **Der größte Boost für die Erkennung leichter Apnoen & Hypopnoen (+5 % bis +10 % Accuracy):** Während `THOR` und `ABDO` bei einer **obstruktiven Apnoe** weiterwackeln (weil der Patient gegen den verschlossenen Rachen ankämpft!), fällt `FLOW` auf **eine glatte Nulllinie**. Damit erkennt selbst das simpelste Modell sofort den echten Atemstopp! | • **Wird `FLOW` deaktiviert:** Das Modell muss aus den weiterlaufenden Brust-/Bauchgurten (`THOR`/`ABDO`) erraten, ob noch Luft durch den Rachen strömt oder nicht. Genau deshalb liegt ein 4-Kanal-Modell ohne `FLOW` (*Alarcón 2023*: `84,3 %`) deutlich unter einem Modell mit `FLOW` (`> 93–95 %`)! | • `FLOW + SPO2` (Dual-Branch CNN): Erreicht **95,91 % Accuracy** (*Peng et al., 2024/2025*). |
| **5. `EEG_C4`**<br>*(Hirnaktivität / 8–30 Hz Arousal-Bandpower)* | Misst **kortikale Weckreaktionen (Arousals: Alpha/Beta-Spikes `>= 3 s`)** am Ende einer Apnoe sowie die Schlaftiefe (`Delta`-Power). | **Starkes Plus bei Grenzfällen (Hypopnoen ohne großen SpO2-Drop) & Filter gegen Wach-Fehlalarme:**<br>1. Manche Patienten wachen schon nach `12 s` Atemverengung kurz auf (EEG-Arousal), *bevor* der `SpO2`-Wert um `3 %` fällt -> nur mit `EEG_C4` als echte Hypopnoe erkennbar!<br>2. Hilft dem Modell, **unregelmäßige Wach-Atmung** von echtem Schlaf zu unterscheiden (**höhere Specificity bei Gesunden**). | • **Wird `EEG_C4` deaktiviert:** Kein Problem für schwere Apnoen (`AHI >= 30`), aber bei gesunden/milden Patienten (`AHI < 15`) steigt die Rate an **False Positives in Wachphasen** (das Hauptproblem in *Alarcón et al., 2023, S. 17*!), und Arousals ohne Entsättigung bleiben unsichtbar. | • Steigert die kombinierte Schlaf-/Event-Erkennung von `~70 %` auf **`79–90 %`** (*Huttunen et al., 2022; Fathima & Ahmed, 2025*). |
| **6. `ECG_AMP`**<br>*(EKG-R-Zacken-Amplitude / EDR)* | Misst über die Schwankung der R-Zacken-Höhe die **abgeleitete Brustkorbatmung (ECG-Derived Respiration, EDR)**. | **Perfekter Backup-Kanal, wenn Brust-/Bauchgurte verrutschen oder fehlen:** Selbst wenn `THOR`, `ABDO` und `FLOW` deaktiviert sind, liefert `ECG_AMP` zusammen mit `HR` eine erstaunlich genaue Ersatz-Atmungskurve direkt aus dem EKG. | • **Wird `ECG_AMP` deaktiviert:** Solange `THOR`, `ABDO` und `FLOW` aktiv sind, ist der Verlust minimal (`< 1 %`). Sind die Atmungsgurte aber deaktiviert (z. B. im `Apnea-ECG`-Datensatz), ist `ECG_AMP` + `HR` überlebenswichtig! | • `HR + EDR` allein erreicht auf *Apnea-ECG* bis zu **93–97 % Acc** (*Peng et al., 2025*). |
| **7. Optional: `POS` & `SNORE`**<br>*(Körperlage & Schnarchmikrofon)* | `POS`: Rückenlage vs. Seitenlage.<br>`SNORE`: Lautes Schnarchen -> **Stille** -> **Explosives Luftschnappen**. | • `POS` liefert wertvolles Kontextwissen (über 50 % aller OSA-Events passieren in **Rückenlage**!).<br>• `SNORE` erkennt den akustischen Wechsel von Schnarchen zu absoluter Stille während des Rachenverschlusses. | • Nicht in jedem Datensatz enthalten, daher ideal als **optional zuschaltbare Bonus-Kanäle** für Datensätze wie `UCDDB` und `ISRUC`. | • `SNORE + SPO2`: **92,45 % Acc** (*Montazeri et al., 2023*). |

#### Zusammenfassung für eure Paper-Hypothese (Die erwartete Qualitäts-Treppe von Setup 1 bis Setup 5):
Wenn ihr in eurem Paper die 5 vordefinierten Sensor-Toggle-Setups nacheinander aktiviert, könnt ihr genau diese **physiologisch begründete Qualitätssteigerung** nachweisen und diskutieren:
1. **`Setup_1` (`SPO2 + HR` – 2 Kanäle):** Solide Basis für schwere Apnoen mit starkem Sauerstoffabfall, aber zeitverzögert und blind für kurze Apnoen ohne großen `SpO2`-Drop (*erwartete Acc: ~78–82 %*).
2. **`Setup_2` (`+ THOR + ABDO` – 4 Kanäle, Alarcón-2023-Baseline):** Deutlicher Sprung in der **Sensitivity**, weil die mechanische Atemanstrengung sofort sichtbar wird (*erwartete Acc: ~84–87 %*).
3. **`Setup_3` (`+ FLOW` – 5 Kanäle):** **Größter Qualitäts-Sprung**, weil der direkte Luftstrom an Nase/Mund den echten Atemstillstand von bloßer Gurtbewegung trennt (*erwartete Acc: ~90–94 %*).
4. **`Setup_4 & 5` (`+ EEG_C4 + ECG_AMP` – 6 bis 7 Kanäle):** Höchste **Specificity & Präzision bei Grenzfällen**, da kortikale Weckreaktionen (EEG-Arousals) erkannt und Wach-Artefakte unterdrückt werden (*erwartete Acc: ~92–96 %*).

---

### 5.4 Variables Sensor-Toggle-System (Sensordaten während Entwicklung & Forschung flexibel aktivieren/deaktivieren)

Du hast dir gewünscht, dass man **verschiedene Sensordaten während des Entwicklungs- und Forschungsprozesses flexibel aktivieren oder deaktivieren** kann. Das hat für eure reine Paper-Forschung zwei riesige Vorteile:
1. **Automatisierte Ablationsstudien-Tabellen fürs Paper:** Ihr könnt in einer einfachen Schleife automatisch testen, wie gut das Modell mit verschiedenen Sensor-Kombinationen (z. B. *„Nur Pulsoximeter“* vs. *„Portable Monitor mit 4 Kanälen“* vs. *„Mit Atemfluss & EEG“*) abschneidet. Das Skript spuckt euch am Ende **fertige CSV-/Excel-/LaTeX-Tabellen** aus, auf die ihr euch im Paper direkt stützen könnt!
2. **Problemloses Kombinieren verschiedener Datensätze:** Falls ein Datensatz (z. B. *SHHS*) einen bestimmten Kanal nicht hat oder ihr die Schnittmenge aus 3 Datensätzen auswerten wollt, aktiviert oder deaktiviert ihr die gewünschten Kanäle einfach in der Konfigurationsliste.

Hier sind die **zwei konkreten, kinderleicht umsetzbaren Methoden**, wie ihr dieses Sensor-Toggle-System in Python für eure Forschung nutzt:

#### Methode 1: Dynamischer `SensorSelector` per Konfigurations-Liste (Für gezielte Experimente & Ablationstabellen)
Alle 7 harmonisierten Standard-Kanäle liegen in einer festen Reihenfolge im gespeicherten Array `X` mit der Dimension `(N_windows, 7, 60)` vor. Über eine einfache Python-Liste `ACTIVE_CHANNELS` (oder das Dictionary `SENSOR_PRESETS`) schaltet ihr beliebige Sensoren an oder aus:

```python
import numpy as np

# Festes Verzeichnis aller 7 Kanäle in unserer vereinheitlichten Pipeline:
ALL_CHANNELS = ["SPO2", "HR", "THOR", "ABDO", "FLOW", "EEG_C4", "ECG_AMP"]

# Vordefinierte Sensor-Profile für unsere Paper-Tabellen (oder völlig frei kombinierbar!):
SENSOR_PRESETS = {
    "Setup_1_Oximeter_2ch":      ["SPO2", "HR"],                                  # Nur Finger-Sensor (Level IV)
    "Setup_2_PortableMon_4ch":   ["SPO2", "HR", "THOR", "ABDO"],                  # Wie Alarcón et al. 2023 (Level III)
    "Setup_3_Respiratory_5ch":   ["SPO2", "HR", "THOR", "ABDO", "FLOW"],          # PM + Nasen-Atemfluss
    "Setup_4_NeuroCardio_6ch":   ["SPO2", "HR", "THOR", "ABDO", "FLOW", "EEG_C4"],# Inkl. Gehirnaktivität (EEG)
    "Setup_5_Full_7ch":          ["SPO2", "HR", "THOR", "ABDO", "FLOW", "EEG_C4", "ECG_AMP"] # Alle 7 Kanäle
}

def select_active_sensors(X_full: np.ndarray, active_channels: list[str]) -> np.ndarray:
    """
    Schneidet aus dem Gesamt-Tensor X_full (N_windows, 7, 60) dynamisch nur die
    aktuell aktivierten Sensoren heraus -> Rückgabe: (N_windows, len(active_channels), 60).
    """
    channel_indices = [ALL_CHANNELS.index(ch) for ch in active_channels]
    return X_full[:, channel_indices, :]

def extract_simple_tabular_features(X_selected: np.ndarray) -> np.ndarray:
    """
    Für Klassisches ML (Random Forest / LightGBM) & statistische Auswertungstabellen:
    Berechnet pro aktiviertem Kanal 5 simple Kennzahlen über das 60s-Fenster:
    [Mittelwert, Standardabweichung, Minimum, Maximum, Max-Min-Spanne (Drop)].
    Aus (N_windows, C_active, 60) wird eine flache Tabelle (N_windows, C_active * 5).
    """
    feat_mean = np.mean(X_selected, axis=2)
    feat_std  = np.std(X_selected, axis=2)
    feat_min  = np.min(X_selected, axis=2)
    feat_max  = np.max(X_selected, axis=2)
    feat_ptp  = feat_max - feat_min  # Peak-to-Peak (z.B. Tiefe des SpO2-Abfalls!)
    return np.concatenate([feat_mean, feat_std, feat_min, feat_max, feat_ptp], axis=1)
```

#### Methode 2: Zero-Masking (`apply_sensor_mask`) für datensatzübergreifendes Training bei fehlenden Kanälen
Wenn ihr mehrere Datensätze gleichzeitig in **einem einzigen Modell** trainieren oder testen wollt, aber Datensatz A (z. B. `UCDDB`) alle 7 Kanäle besitzt und Datensatz B (z. B. `SHHS2`) nur 4 Kanäle hat, könnt ihr die fehlenden/deaktivierten Kanäle einfach mit `0.0` maskieren:

```python
def apply_sensor_mask(X_full: np.ndarray, active_channels: list[str]) -> np.ndarray:
    """
    Behält die feste Form (N_windows, 7, 60) bei, setzt aber alle
    NICHT in active_channels enthaltenen Sensoren sauber auf 0.0.
    Ideal, um ein trainiertes Modell ohne Neutraining mit deaktivierten Sensoren zu testen!
    """
    X_masked = X_full.copy()
    for idx, ch_name in enumerate(ALL_CHANNELS):
        if ch_name not in active_channels:
            X_masked[:, idx, :] = 0.0
    return X_masked
```

---

## 6. Simples Modell-Design (Binär: Apnoe vs. Keine Apnoe) & Automatisierte Auswertungs-Pipeline (Tabellen & Plots fürs Paper)

### 6.1 Zwei einfache, bewährte Modell-Architekturen für euren binären Apnoe-Test

Da euer Modell **nur binär auf `Apnoe (1)` vs. `Keine Apnoe (0)` testen muss** und ihr keine Endanwendung bauen müsst, könnt ihr euch zu 100 % auf saubere, leicht auswertbare Modelle konzentrieren. Wir empfehlen euch, im Team **zwei sehr einfache, komplementäre Modelle** auf denselben Daten zu vergleichen (damit deckt ihr gleichzeitig *Phase 1: Classic ML vs. Deep Learning* aus den Vorlesungsfolien perfekt ab und habt doppelt so viel Stoff für eure Ergebnistabellen!):

#### Modell A (Ultra-Simpel & Extrem Schnell): Random Forest / LightGBM auf statistischen Fenster-Features
* **Input:** Für jeden aktivierten Sensor die 5 simplen Kennzahlen aus `extract_simple_tabular_features()` (`Mean`, `Std`, `Min`, `Max`, `Peak-to-Peak-Drop` pro 60s-Fenster). Bei 4 aktivierten Sensoren sind das gerade einmal **20 Zahlen pro 60s-Fenster**!
* **Klassifikator:** `RandomForestClassifier(n_estimators=200, class_weight="balanced")` aus `scikit-learn` oder `LGBMClassifier()`.
* **Vorteile für euer Paper:**
  * Trainiert auf 50.000 Fenstern in **weniger als 5 Sekunden** auf jedem normalen Laptop (keine GPU nötig!).
  * Liefert über `model.feature_importances_` sofort eine **fertige Prozent-Tabelle für das Paper**, welcher aktivierte Sensor wie stark zur Apnoe-Erkennung beigetragen hat!

#### Modell B (Simples 1D-CNN analog zu Alarcón et al., 2023): Zeitreihen-Klassifikator
* **Input:** Das 60-Sekunden-Zeitfenster der aktivierten Kanäle `(Batch, C_active, 60)` bei `1 Hz`.
* **Architektur (Kompaktes 4-Layer 1D-CNN mit Global Average Pooling):**
  ```text
  Input: (Batch_Size, C_active, 60 Sekunden)
     |
     v
  [Block 1] Conv1D(64 Filter, Kernel=7, Padding='same') -> BatchNorm -> ReLU -> Dropout(0.3)
     |
     v
  [Block 2] Conv1D(128 Filter, Kernel=5, Padding='same') -> BatchNorm -> ReLU -> Dropout(0.3)
     |
     v
  [Block 3] Conv1D(128 Filter, Kernel=3, Padding='same') -> BatchNorm -> ReLU -> Dropout(0.3)
     |
     v
  [Global Average Pooling 1D] -> mittelt über die 60 Sekunden -> Vektor (Batch_Size, 128)
     |
     v
  [Output] Dense(1, Activation='sigmoid') -> Wahrscheinlichkeit p(Apnoe) ∈ [0.0, 1.0]
           (p >= 0.5 -> 1 = Apnoe | p < 0.5 -> 0 = Keine Apnoe)
  ```
* **Vorteile für euer Paper:**
  * Direkt vergleichbar mit den Ergebnissen im Paper eures Betreuers (*Alarcón et al., 2023*).
  * Durch das `GlobalAveragePooling1D` vor dem Output-Layer könnt ihr für das Paper eine beeindruckende **1D-Grad-CAM-Grafik** plotten, die auf einem 60s-Zeitfenster zeigt, welche Sekunden das CNN als Apnoe-Auslöser markiert hat!

---

### 6.2 Der automatisierte „Paper-Tabellen & Plot-Generator“ (Statt einer GUI!)

Da am Ende **keine App/GUI**, sondern **belastbare Forschungsdaten, Auswertungstabellen und Plots für euer Paper** entstehen sollen, ist das Endprodukt eurer Pipeline ein sauberes **Benchmark- & Auswertungs-Skript (`run_experiments.py`)**. Es durchläuft automatisch alle Sensor-Toggles und Datensätze und speichert die fertigen Tabellen (`.csv`, `.xlsx`, `.tex`) und Abbildungen (`.png`/`.pdf` in 300 DPI):

```text
+---------------------------------------------------------------------------------------------------+
|                AUTOMATISIERTE FORSCHUNGS- & AUSWERTUNGS-PIPELINE (run_experiments.py)             |
|                     Keine GUI nötig • Erzeugt direkt alle Tabellen & Plots fürs Paper             |
+---------------------------------------------------------------------------------------------------+
| SCHRITT 1: Iteriere automatisch über alle Datensätze (z.B. UCDDB, ISRUC, Apnea-ECG, Kombiniert)   |
|            UND über alle Sensor-Toggle-Setups (Setup 1 [2ch] bis Setup 5 [7ch])                   |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
| SCHRITT 2: Trainiere & Evaluiere pro Kombination:                                                 |
|            • Modell A: Random Forest / LightGBM (auf 5 Statistik-Features pro aktivem Kanal)      |
|            • Modell B: 1D-CNN (auf den 60s 1-Hz-Zeitreihen der aktiven Kanäle)                    |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
| SCHRITT 3: Automatischer Export der 4 Ergebnis-Tabellen (.csv / .xlsx / LaTeX .tex fürs Paper)    |
|                                                                                                   |
|  1. table1_dataset_summary.csv:                                                                   |
|     Datensatz | Patienten (N) | 60s-Fenster gesamt | Apnoe-Fenster (%) | Mittlerer AHI ± SD       |
|                                                                                                   |
|  2. table2_sensor_ablation_results.csv (EURE WICHTIGSTE PAPER-TABELLE!):                          |
|     Setup-Name | Aktive Sensoren | Modell | Accuracy | Sensitivity | Specificity | F1 | AUC | LR+ |
|                                                                                                   |
|  3. table3_cross_dataset_results.csv:                                                             |
|     Train-Datensatz -> Test-Datensatz | Accuracy | Sensitivity | Specificity | F1-Score | AUROC   |
|                                                                                                   |
|  4. table4_sensor_feature_importance.csv:                                                         |
|     Sensor-Kanal | Wichtigstes Merkmal (z.B. SpO2_Drop, Flow_Min, HR_Std) | Beitrag in %          |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
| SCHRITT 4: Automatischer Export der publikationsreifen Paper-Plots (Matplotlib/Seaborn, 300 DPI)  |
|  • fig2_roc_and_ablation.pdf: ROC-Kurven aller Sensor-Setups + Balkendiagramm (1D-CNN vs. RF)     |
|  • fig3_gradcam_and_importance.pdf: 60s-Apnoe-Fenster mit 1D-Grad-CAM-Heatmap + Sensor-Ranking    |
+---------------------------------------------------------------------------------------------------+
```

Hier ist das konkrete **Vorlage-Skript (`run_experiments.py`)**, mit dem ihr genau diese Auswertungstabellen in Python automatisch erzeugt:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Berechnet alle wissenschaftlichen Metriken für unsere Paper-Tabellen (wie Alarcón 2023)."""
    acc  = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1)          # Sensitivity (True Positive Rate)
    spec = recall_score(y_true, y_pred, pos_label=0)          # Specificity (True Negative Rate)
    prec = precision_score(y_true, y_pred, zero_division=0)   # Precision (PPV)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    lr_plus = sens / (1.0 - spec + 1e-6)                      # Positive Likelihood Ratio (Ziel >= 5.0)
    return {
        "Accuracy (%)": round(acc * 100, 2),
        "Sensitivity (%)": round(sens * 100, 2),
        "Specificity (%)": round(spec * 100, 2),
        "Precision (%)": round(prec * 100, 2),
        "F1-Score (%)": round(f1 * 100, 2),
        "AUC (%)": round(auc * 100, 2),
        "LR+": round(lr_plus, 2)
    }

# Automatische Schleife über alle Sensor-Toggles -> erzeugt direkt die Paper-Tabelle!
results_rows = []
for setup_name, active_chs in SENSOR_PRESETS.items():
    X_train_sel = extract_simple_tabular_features(select_active_sensors(X_train_full, active_chs))
    X_test_sel  = extract_simple_tabular_features(select_active_sensors(X_test_full, active_chs))
    
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_train_sel, y_train)
    
    y_pred = clf.predict(X_test_sel)
    y_prob = clf.predict_proba(X_test_sel)[:, 1]
    
    metrics = evaluate_predictions(y_test, y_pred, y_prob)
    results_rows.append({
        "Setup": setup_name,
        "Active Sensors": ", ".join(active_chs),
        "N_Channels": len(active_chs),
        "Model": "Random Forest",
        **metrics
    })

df_ablation = pd.DataFrame(results_rows)
df_ablation.to_csv("table2_sensor_ablation_results.csv", index=False)
df_ablation.to_latex("table2_sensor_ablation_results.tex", index=False) # Direkt für Overleaf/LaTeX!
```

---

## 7. Vereinfachte Arbeitsverteilung für 6 Personen (WP 1 – WP 6) & Entspannter Zeitplan bis Ende Dezember 2026

### 7.1 Die 6 Work Packages (WP 1 – WP 6) für den reinen Forschungs- & Paper-Durchlauf (Ohne GUI!)

Da keine Endanwendung gebaut werden muss, konzentrieren sich alle 6 Teammitglieder vollständig auf die **Daten-Pipeline, die Experimente, die Auswertungstabellen/Plots und das Schreiben des KES-Papers**. Jede Person hat ein klar abgegrenztes Arbeitspaket für ihren **15-Minuten-Einzelvortrag (40 %)** und ihren **Teil im 8–10-seitigen Paper (60 %)**:

* **Person 1 – WP 1: Multi-Dataset-Adapter & Kanonische Daten-Pipeline (Team Leader)**
  * *Aufgaben:* Herunterladen von mindestens 2–3 freien Datensätzen (z. B. `UCDDB`, `ISRUC-Sleep`, `Apnea-ECG` / `CinC 2018`, optional `SHHS/MESA`); Schreiben der kurzen Python-Ladefunktionen (Stufe 1 & 2 der Pipeline), die die Signale einlesen, auf die 7 Standard-Namen (`SPO2`, `HR`, `THOR`, `ABDO`, `FLOW`, `EEG_C4`, `ECG_AMP`) mappen und auf `1 Hz` bringen; Erstellen von **`table1_dataset_summary.csv`** (Statistik über alle genutzten Datensätze).
  * *Paper-Kapitel & 15-Min.-Vortrag:* Einleitung, Überblick über öffentliche Schlaf-Datenbanken, Unterschiede zwischen den Datensätzen und Aufbau des Multi-Dataset-Adapters.
* **Person 2 – WP 2: Signal-Vorverarbeitung, 60s-Windowing & Variables Sensor-Toggle-Modul**
  * *Aufgaben:* Implementierung von Stufe 3 & 4 der Pipeline: Artefakt-Filterung (`SpO2` & `HR` Interpolation), Vergleich zweier einfacher Normalisierungen (*Ohne Normierung vs. Fensterweiser Z-Score vs. Patienten-IQR-Skalierung*), 60s-Fenster-Segmentierung mit binärem Label (`0 = Keine Apnoe`, `1 = Apnoe` ab `>= 10 s`) sowie Bau des **`SensorSelector` / Sensor-Toggle-Moduls** (`select_active_sensors` & `apply_sensor_mask`).
  * *Paper-Kapitel & 15-Min.-Vortrag:* Biophysik der Schlafsensoren (Kapitel 4), Einfluss von Artefakt-Filterung und Normalisierung (Ablationstabelle analog zu Model 1–4 in *Alarcón et al., 2023*) und Funktionsweise des flexiblen Sensor-Toggles.
* **Person 3 – WP 3: Klassisches Machine Learning (Random Forest / LightGBM) & Statistische Feature-Analyse**
  * *Aufgaben:* Berechnung der 5 einfachen statistischen Merkmale pro Kanal und 60s-Fenster (`Mean`, `Std`, `Min`, `Max`, `Peak-to-Peak`); Training und Auswertung von **Random Forest** und **LightGBM** (`0` vs. `1`); Erstellung von **`table4_sensor_feature_importance.csv`** (Ranking, welche statistischen Merkmale und Sensoren den höchsten Informationsgewinn liefern).
  * *Paper-Kapitel & 15-Min.-Vortrag:* Grundlagen Classic ML vs. Deep Learning, Erklärung der statistischen Fenster-Features, Ergebnisse der Baum-Modelle und Analyse der Feature Importance.
* **Person 4 – WP 4: Simples 1D-CNN (Deep Learning analog zu Alarcón et al., 2023)**
  * *Aufgaben:* Aufbau und Training des schlanken **1D-CNN mit Global Average Pooling** (3–4 `Conv1D`-Blöcke + `Sigmoid`-Output) direkt auf den `1 Hz` 60-Sekunden-Zeitreihen; Vergleich von Trainings-Setups (z. B. balanciertes `1:1` Training vs. unbalancierte Testdaten) sowie Gegenüberstellung **1D-CNN vs. Random Forest/LightGBM**.
  * *Paper-Kapitel & 15-Min.-Vortrag:* Funktionsweise von 1D-CNNs auf physiologischen Zeitreihen, Netzwerkarchitektur, Hyperparameter, Trainingsverlauf (Loss/Accuracy-Kurven) und Vergleich mit Classic ML.
* **Person 5 – WP 5: Systematische Sensor-Toggle-Experimente (Ablationsstudie) & Multi-Dataset-Evaluation**
  * *Aufgaben:* Durchführung der großen **Experiment-Serie für das Paper** durch systematisches Aktivieren/Deaktivieren von Sensoren über das Toggle-System (*Setup 1: Nur `SpO2 + HR`* bis *Setup 5: Alle 7 Kanäle*) sowie Auswertung auf den einzelnen und den **kombinierten Datensätzen** (In-Dataset vs. Cross-Dataset); Erzeugung der beiden zentralen Ergebnistabellen **`table2_sensor_ablation_results.csv`** und **`table3_cross_dataset_results.csv`**.
  * *Paper-Kapitel & 15-Min.-Vortrag:* Vorstellen der zentralen Ergebnistabellen und ROC-Kurven: *„Wie viele und welche Sensoren braucht man wirklich für eine gute Apnoe-Erkennung, und wie stark verbessert das Zusammenlegen mehrerer Datensätze die Generalisierung?“*
* **Person 6 – WP 6: Patienten-Level AHI-Auswertung, Explainable AI (1D-Grad-CAM Plots) & Wissenschaftliche Diskussion / Ausblick**
  * *Aufgaben:* Berechnung des geschätzten **Patienten-AHI** (`Erkannte Apnoe-Fenster / Aufnahmedauer in h`, Korrelation `R²`, MAE und Schweregrad-Konfusionsmatrix `AHI < 5, 5–15, 15–30, >= 30` genau wie in *Alarcón et al., 2023*); Generierung der publikationsreifen **1D-Grad-CAM-Beispielgrafiken** (die auf einem 60s-Fenster zeigen, warum das Modell Apnoe erkannt hat); Ausformulieren des **Diskussions-, Literaturvergleichs- und Ausblick-Kapitels** im Paper (inkl. Zusatz A [Apnoe-Subtypen] und Zusatz B [Schlafphasen-Kopplung] aus Kapitel 8).
  * *Paper-Kapitel & 15-Min.-Vortrag:* Patienten-Level AHI-Evaluation, Explainable AI (Grad-CAM-Fallbeispiele für True Positives & False Positives), Vergleich eurer Tabelle mit der SOTA-Literatur und wissenschaftlicher Ausblick (*Future Work*).

---

### 7.2 Entspannter 10-Wochen-Zeitplan (Ziel: Komplett fertig vor Weihnachten / Ende Dezember 2026!)

* **Woche 1–2 (19. Okt. – 01. Nov. 2026) – Setup & Datensätze sichten:**
  * Rollen (WP 1–6) verteilen, Git-Repository und gemeinsamen Zotero-Ordner anlegen.
  * `UCDDB` (`25` PSGs) und `Apnea-ECG` / `ISRUC` herunterladen; erstes Einlesen einer `.edf`-Datei in Python ausprobieren.
* **Woche 3–4 (02. Nov. – 15. Nov. 2026) – Einheitliche 60s-Pipeline & Sensor-Toggle steht:**
  * **WP 1 & WP 2:** Die Datensätze werden automatisiert in das einheitliche `(N_windows, 7_Kanäle, 60_Sekunden)`-Format mit binärem Label `y ∈ {0, 1}` konvertiert. Das `SensorSelector`-Toggle-Modul funktioniert.
  * **WP 3 & WP 4:** Erste Trainingsläufe von Random Forest und 1D-CNN auf einem ersten Datensatz (`UCDDB`).
* **Woche 5–6 (16. Nov. – 29. Nov. 2026) – Automatisierte Sensor-Ablation & Multi-Dataset-Läufe:**
  * **WP 3, WP 4 & WP 5:** `run_experiments.py` läuft über alle 5 Sensor-Toggle-Setups und Datensatz-Kombinationen und exportiert die ersten CSV-Auswertungstabellen.
  * **WP 6:** Skript für Patienten-AHI-Auswertung (`R²`) und 1D-Grad-CAM-Plot steht.
  * **Paper:** Aufsetzen des **KES-Templates**, Schreiben von *Introduction, Related Work* und *Methods*.
* **Woche 7–8 (30. Nov. – 13. Dez. 2026) – Finale Auswertungstabellen & die 5 Paper-Grafiken fertigstellen:**
  * **WP 5 & WP 6:** Finalisierung der **exakt 5 erlaubten Tabellen/Abbildungen** für das KES-Paper aus den generierten CSV-Daten.
* **Woche 9–10 (14. Dez. – 23./30. Dez. 2026) – Paper-Feinschliff & Folien vor Silvester:**
  * Gemeinsames Fertigschreiben von *Results, Discussion & Future Work* (Kapitel 8 liefert euch dafür bereits die kompletten Argumente!).
  * Erstellen der Präsentationsfolien (6 × 15 Min.).
* **Januar 2027 (Entspannter Puffer):**
  * Optionales Feedback vom Betreuer einarbeiten und 1–2 Probedurchläufe für den Vortrag.

---

### 7.3 Euer Plan für die exakt 5 erlaubten Grafiken/Tabellen im KES-Paper
Da das KES-Paper **strikt maximal 5 grafische Elemente (Tabellen + Abbildungen zusammen!)** enthalten darf, stützt ihr euer Paper auf genau diese **2 Tabellen und 3 Abbildungen**, die euer Skript direkt ausgibt:
1. **Figure 1 (Methodik-Blockdiagramm):** Übersichtliches Flussdiagramm eurer Multi-Dataset-Forschungspipeline (Verschiedene Roh-Datensätze -> 1-Hz-Harmonisierung & 60s-Windowing -> Variables Sensor-Toggle-Modul `ACTIVE_CHANNELS` -> Klassisches ML vs. 1D-CNN -> Segment- & Patienten-AHI-Auswertung).
2. **Table 1 (Datensatz- & Sensor-Ablations-Haupttabelle):** Kombinierte Ergebnistabelle eurer 5 Sensor-Toggle-Setups (*Setup 1: `SpO2+HR`* bis *Setup 5: Alle 7 Kanäle*) für **Random Forest/LightGBM vs. 1D-CNN** über die ausgewerteten Datensätze hinweg (mit Accuracy, Sensitivity, Specificity, F1, AUC und `LR+`).
3. **Figure 2 (Multi-Panel Auswertungs-Plot):**
   * *Links:* ROC-Kurven der verschiedenen aktivierten Sensor-Kombinationen.
   * *Rechts:* AHI-Korrelationsplot (`AHI_predicted` vs. `AHI_true` pro Patient mit `R²`-Wert) bzw. Cross-Dataset-Balkendiagramm.
4. **Figure 3 (Erklärbarkeit: 1D-Grad-CAM & Sensor-Feature-Importance):**
   * *Links:* Ein 60-Sekunden-Apnoe-Fenster mit den aktivierten Sensorkurven (`SpO2`, `HR`, `THOR`, `ABDO`, `FLOW`) und hinterlegter **1D-Grad-CAM-Heatmap**, die zeigt, welche Sekunden für die Klassifikation ausschlaggebend waren.
   * *Rechts:* Balkendiagramm der prozentualen Sensor-Wichtigkeit (Feature Importance).
5. **Table 2 (Vergleich mit der State-of-the-Art-Literatur):** Gegenüberstellung eurer besten Sensor-Setups mit den Ergebnissen aus *Alarcón et al. (2023)*, *Peng et al. (2024/2025)* und *Gutiérrez-Tobal et al. (2021)*.

---

## 8. Zusatz & Ausblick: Was wäre notwendig für (A) Verschiedene Apnoe-Typen und (B) Einbezug eines Schlafphasen-Modells?

Dieses Kapitel beantwortet konkret deine Zusatzfrage: **Was müsste man an unserem simplen Basis-System ändern oder ergänzen, wenn man (A) statt nur binär (`Apnoe` vs. `Keine Apnoe`) auf verschiedene Apnoe-Typen testen will, oder (B) das Modell durch die Kopplung mit einem Schlafphasen-Modell verbessern möchte?**

> **Praxis-Tipp für euer Team:**
> Ihr habt hier zwei Möglichkeiten:
> 1. Entweder ihr nutzt dieses Kapitel direkt als **fundiertes „Discussion & Future Work“-Kapitel in eurem KES-Paper** (um dem Professor zu zeigen, dass ihr genau wisst, wie man das simple System wissenschaftlich ausbaut),
> 2. **oder** eine Person im Team (z. B. WP 3 oder WP 4) probiert eine dieser beiden Erweiterungen als **kleines Bonus-Experiment** auf einem Datensatz (z. B. `UCDDB`) aus!

---

### 8.1 Erweiterung A: Von Binär (`Apnoe vs. Keine Apnoe`) zu Multi-Klassen (`Normal`, `OSA`, `CSA`, `MSA`, `Hypopnoe`)

Wenn euer Modell nicht nur erkennen soll, *dass* eine Atmungsstörung vorliegt, sondern auch **welcher Subtyp** (`0 = Normal`, `1 = Obstruktive Apnoe [OSA]`, `2 = Zentrale Apnoe [CSA]`, `3 = Hypopnoe`), sind **genau 4 konkrete Anpassungen** an Daten, Sensoren, Features und Modell nötig:

#### 1. Zwingende Sensor-Voraussetzungen im Sensor-Toggle (`ACTIVE_CHANNELS`)
Mit reinem `SpO2 + HR` ist eine Trennung von OSA und CSA **physikalisch fast unmöglich**, weil beide im Blut zu einem Sauerstoffabfall und einer Herzfrequenz-Reaktion führen. Für die Subtypen-Trennung müssen im Sensor-Toggle zwingend folgende Kanäle aktiviert sein:
* **`FLOW` (Oronasaler Thermistor / Nasaldruck):** Nötig, um eine **komplette Apnoe (`>= 90 %` Amplitudenabfall)** von einer bloßen **Hypopnoe (`30 % – 89 %` Amplitudenabfall)** zu unterscheiden.
* **`THOR` (Brustgurt) UND `ABDO` (Bauchgurt):** Zwingend nötig, um **OSA** von **CSA** zu unterscheiden:
  * Bei **OSA** arbeiten Brust- und Bauchgurt während des Atemstillstands weiter – und zwar **gegenphasig (paradoxe Schaukelatmung: Brust zieht ein, während Bauch sich vorwölbt)**.
  * Bei **CSA** fällt der zentrale Atemantrieb im Gehirn aus -> **sowohl `THOR` als auch `ABDO` sind während des Atemstillstands komplett flach (Amplitude $\approx 0$)**!

#### 2. Höhere Abtastrate für die Atmungskanäle (`10 Hz` statt `1 Hz`)
* Ein normaler Atemzug dauert ca. `3–4 Sekunden` (`~0,25–0,33 Hz`). Bei einer starken Heruntertastung auf `1 Hz` (wie im simplen Basis-Modell) erkennt man die **Gegenphasigkeit (Paradoxie)** zwischen Brust- und Bauchgurt sowie die abgeflachte Kurvenform (*Flow Limitation*) einer Hypopnoe nur noch schlecht.
* **Notwendige Änderung:** In Stufe 2 der Pipeline werden `THOR`, `ABDO` und `FLOW` auf **`10 Hz`** (`600` Werte pro 60s-Fenster) statt `1 Hz` abgetastet.

#### 3. Das entscheidende physikalische Zusatz-Feature für Klassisches ML & CNNs: Die Thorako-Abdominale Phasen-Korrelation `r(THOR, ABDO)`
Um dem Modell die Unterscheidung zwischen OSA und CSA auf dem Silbertablett zu servieren, berechnet man in jedem 60s-Fenster (bzw. in gleitenden 10s-Subfenstern) zwei zusätzliche Zahlen:
1. **Kreuzkorrelation zwischen Brust- und Bauchgurt (`r_thor_abdo`):**
   $$\text{corr}(\text{THOR}, \text{ABDO}) = \frac{\sum (x_{\text{thor}} - \bar{x})(x_{\text{abdo}} - \bar{y})}{\sqrt{\sum (x_{\text{thor}} - \bar{x})^2 \sum (x_{\text{abdo}} - \bar{y})^2}} \in [-1, +1]$$
   * **Normale Atmung:** Brust und Bauch heben sich synchron -> **`r_thor_abdo` ist stark positiv (`+0,6 bis +0,95`)**.
   * **Obstruktive Apnoe (OSA):** Paradoxe Schaukelatmung gegen den verschlossenen Rachen -> **`r_thor_abdo` wird negativ (`-0,2 bis -0,9`)**!
   * **Zentrale Apnoe (CSA):** Keine Bewegung in Brust und Bauch -> **Varianz/Standardabweichung `std(THOR)` und `std(ABDO)` fällt auf nahezu `0`**!
2. **Relativer Airflow-Drop (`drop_flow`):**
   Verhältnis der minimalen 10s-Amplitude im Fenster zur 120s-Baseline davor (`<= 0,10` -> Apnoe; `0,11 bis 0,70` + `SpO2-Drop >= 3 %` -> Hypopnoe).

#### 4. Drei kleine Code-Änderungen am Modell & Training (Wegen extremer Class Imbalance!)
In echten Schlaflabor-Daten sind die Klassen extrem ungleich verteilt (z. B. `82 % Normal`, `10 % OSA`, `6 % Hypopnoe`, aber nur `1–2 % CSA` und `< 0,5 % MSA`). Wenn man ein normales Netz trainiert, ignoriert es `CSA` komplett.
* **Änderung 1 (Label-Vektor):** Statt `y ∈ {0, 1}` speichert Stufe 4 der Pipeline `y ∈ {0: Normal, 1: OSA, 2: CSA, 3: Hypopnoe}` (Datensätze mit diesen exakten Labels: `UCDDB`, `CinC 2018`, `ISRUC`, `SHHS`, `MESA`, `PSG-Audio`).
* **Änderung 2 (Output-Layer):** Im 1D-CNN wird `Dense(1, activation='sigmoid')` ersetzt durch `Dense(4, activation='softmax')`.
* **Änderung 3 (Loss-Funktion mit Klassengewichtung / Focal Loss):**
  * Beim Random Forest / LightGBM reicht ein einziger Parameter: `class_weight="balanced"`.
  * Beim 1D-CNN ersetzt man `BinaryCrossentropy` durch **Weighted Cross-Entropy** (gewichtet umgekehrt proportional zur Klassenhäufigkeit $w_c = \frac{N_{\text{total}}}{4 \cdot N_c}$) oder **Multi-Class Focal Loss** ($\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ mit $\gamma = 2,0$), damit das Modell gezwungen wird, die seltenen `CSA`- und `Hypopnoe`-Fenster genauso ernst zu nehmen wie `OSA`!

---

### 8.2 Erweiterung B: Verbesserung des Apnoe-Modells & der AHI-Berechnung durch Einbezug eines Schlafphasen-Modells (`Wake, N1, N2, N3, REM`)

Warum ist der Einbezug eines Schlafphasen-Modells die **wissenschaftlich wirkungsvollste Verbesserung** des Ansatzes von *Alarcón et al. (2023)*?
Es gibt dafür **drei direkte Gründe**, die sich in **3 Ausbaustufen (von super einfach bis fortgeschritten)** umsetzen lassen:

```text
+---------------------------------------------------------------------------------------------------+
|      WIE EIN SCHLAFPHASEN-MODELL (W, N1, N2, N3, REM) DEN APNOE-DETEKTOR VERBESSERT               |
+---------------------------------------------------------------------------------------------------+
         |                                       |                                       |
         v                                       v                                       v
+-------------------------------+   +-----------------------------------+   +-----------------------+
| STUFE 1: WACH-FILTER & TST    |   | STUFE 2: SCHLAFPHASE ALS INPUT-   |   | STUFE 3: KLINISCHE    |
| (Post-Processing / AHI-Fix)   |   | FEATURE FÜR DEN APNOE-KLASSIFIK.  |   | PHÄNOTYP-DIAGNOSE     |
+-------------------------------+   +-----------------------------------+   +-----------------------+
| • Problem ohne Schlafphasen:  |   | • Biologisches Vorwissen:         |   | • Trennt automatisch: |
|   1. Im Wachzustand (W) atmet |   |   - In REM (R) sind Apnoen extrem |   |   - REM-AHI vs.       |
|      der Mensch unregelmäßig  |   |     häufig & lang (Muskelatonie!) |   |     NREM-AHI          |
|      (Sprechen, Drehen, Husten|   |   - In Tiefschlaf (N3) sind       |   |   - Wahre Schlaf-     |
|      -> erzeugt False-Positive|   |     Apnoen biologisch sehr selten!|   |     effizienz (SE %)  |
|      Apnoe-Alarme!).          |   | • Lösung:                         |   |   - Schlaf-Fragmentie-|
|   2. AHI wird durch Aufnahme- |   |   Übergib die erkannte Schlafphase|   |     rung durch Apnoen |
|      dauer (TRT=8h) statt     |   |   (z.B. One-Hot [W,N1,N2,N3,REM]  |   |     (% N1 vs. % N3)   |
|      echte Schlafzeit (TST)   |   |   oder EEG-Bandpower Delta/Beta)  |   |                       |
|      geteilt!                 |   |   als Zusatz-Spalten direkt in    |   |                       |
| • Lösung durch Stufe 1:       |   |   den Apnoe-Klassifikator!        |   |                       |
|   - Streiche alle Apnoe-Alarme|   |   -> Steigert F1 & Spezifität!    |   |                       |
|     in Wach-Epochen (State=W) |   +-----------------------------------+   +-----------------------+
|   - Teile durch echte TST!    |
+-------------------------------+
```

#### Ausbaustufe 1 (Am einfachsten – Reines Post-Processing ohne Änderung am Apnoe-Modell!): Wach-Filterung & Wahre `TST`-Berechnung
In *Alarcón et al. (2023, S. 17)* hatten die Autoren zwei große Probleme bei der AHI-Berechnung:
1. **Fehlalarme im Wachzustand (niedrige Spezifität bei Gesunden):** Wenn ein Patient im Bett noch 90 Minuten wach liegt, liest, sich umdreht oder seufzt, schwanken Brust-/Bauchgurt und Herzfrequenz stark. Ein reines Apnoe-Modell ohne Schlaf-Wach-Erkennung hält diese Wach-Unregelmäßigkeiten fälschlicherweise für Apnoen!
2. **Falscher Nenner im AHI (`TRT` statt `TST`):** Sie teilten die Apnoe-Anzahl durch die gesamte Aufnahmedauer (`TRT = 8,0 h`), obwohl viele Patienten nur `5,5 bis 6,5 Stunden` wirklich geschlafen haben (`TST`).

**Was wäre für Ausbaustufe 1 nötig?**
* Ihr lasst parallel zu eurem unveränderten binären Apnoe-Modell ein **Schlafphasen-Modell** (oder zumindest einen **Schlaf-Wach-Klassifikator**) über das `EEG_C4`-Signal (bzw. falls kein EEG aktiv ist, über `HRV + Bewegung/Atmung`) laufen, das für jedes Fenster vorhersagt: `sleep_stage ∈ {0: Wake, 1: N1, 2: N2, 3: N3, 4: REM}`.
* **Die 2 simplen Korrektur-Regeln im Code (5 Zeilen Python):**
  1. **Wach-Maskierung:** Jedes 60s-Fenster, in dem das Schlafphasen-Modell `Wake (0)` erkennt, wird im Apnoe-Output automatisch auf **`0 (Keine Apnoe)`** gesetzt (`y_pred_clean = np.where(sleep_stage == 0, 0, y_pred_apnea)`). *Alle Wach-Fehlalarme verschwinden sofort!*
  2. **Exakte AHI-Formel:** Die wahre Gesamtschlafzeit in Stunden ist `TST_hours = np.sum(sleep_stage != 0) / 60.0`. Der korrigierte AHI lautet dann:
     $$\text{AHI}_{\text{TST}} = \frac{\sum y_{\text{pred\_clean}}}{\text{TST}_{\text{hours}}}$$

#### Ausbaustufe 2 (Feature-Level-Kopplung): Schlafphase als direktes Input-Merkmal für das Apnoe-Modell
Wie wir in Kapitel 3.2 gesehen haben, hängt die Wahrscheinlichkeit einer Schlafapnoe massiv vom aktuellen Schlafstadium ab:
* Im **Tiefschlaf (`N3`)** ist die Atmung extrem stabil -> fast keine Apnoen.
* Im **REM-Schlaf (`R`)** ist die Rachenmuskulatur komplett gelähmt -> höchste Apnoe-Rate und stärkste Sauerstoffabfälle.
* Im **Leichtschlaf (`N1` / `N2`)** führt jedes Apnoe-Ende zu einem kurzen Rückfall in `N1` oder `Wake`.

**Was wäre für Ausbaustufe 2 nötig?**
* Da ein 60-Sekunden-Apnoe-Fenster genau **zwei 30-Sekunden-Schlafstadien-Epochen** (`Epoch_1` von `0–30 s` und `Epoch_2` von `30–60 s`) umfasst, übergibt das Schlafphasen-Modell seine Vorhersage direkt als **zusätzliche Input-Features** an den Apnoe-Klassifikator:
  * **Für euren Random Forest / LightGBM:** Ihr hängt einfach die **5 Klassen-Wahrscheinlichkeiten** des Schlafphasen-Modells (`[P(W), P(N1), P(N2), P(N3), P(REM)]`) sowie ein Flag `stage_transition = (Stage_Epoch1 != Stage_Epoch2)` (zeigt an, ob der Patient mitten in der Minute durch ein Arousal das Schlafstadium gewechselt hat!) als 6 zusätzliche Spalten an eure Feature-Tabelle an!
  * **Für euer 1D-CNN:** Ihr fügt im Sensor-Toggle-System einfach einen zusätzlichen virtuellen Kanal `"SLEEP_STAGE"` (oder die 5 Wahrscheinlichkeits-Spuren) auf `1 Hz` hinzu.

#### Wie kommt ihr ohne großen Aufwand an ein fertiges Schlafphasen-Modell? (2 Optionen)
Falls ihr diese Kopplung tatsächlich im Code testen wollt, müsst ihr das Schlafphasen-Modell **nicht einmal von Grund auf neu erfinden**:
* **Option 1 (Fertige Open-Source-Bibliothek `YASA` – 4 Zeilen Code!):**
  Das in eurer Literatur-Matrix (Paper #11, *Vallat & Walker, 2021, eLife*) aufgeführte Open-Source-Paket **`yasa`** (`pip install yasa`) bringt einen bereits auf über 3.000 NSRR-Nächten vortrainierten Schlafphasen-Klassifikator mit, der mit einem einzigen `EEG`-Kanal (`C4-A1`) in **3 Sekunden für eine ganze Nacht** alle 5 Schlafstadien (`W, N1, N2, N3, R`) vorhersagt:
  ```python
  import yasa
  # raw ist das mit MNE geladene EDF-Objekt eures Patienten:
  sls = yasa.SleepStaging(raw, eeg_name="C4-A1")
  sleep_stages_30s = sls.predict()        # Liefert ['W', 'N1', 'N2', 'N3', 'R'] pro 30s!
  stage_probs_30s  = sls.predict_proba()  # Liefert die Wahrscheinlichkeiten pro Stadium!
  ```
* **Option 2 (Eigenes Mini-Schlafphasen-Modell auf EEG-Bandpower):**
  Ihr berechnet in eurer Pipeline pro 30s-/60s-Fenster mit `scipy.signal.welch` die 5 relativen EEG-Frequenzbänder aus Kapitel 4.2 (`Delta 0,5–4 Hz`, `Theta 4–8 Hz`, `Alpha 8–12 Hz`, `Sigma 11–16 Hz`, `Beta 13–30 Hz`) und trainiert darauf einen zweiten kleinen Random Forest / LightGBM auf die in `UCDDB` und `ISRUC` bereits mitgelieferten Schlafstadien-Labels (`0:W, 1:N1, 2:N2, 3:N3, 4:REM`).

---

### 8.3 Zusammenfassende Checkliste: Basis-Projekt vs. Optionale Erweiterungen

| Komponente | Euer Simples Basis-Projekt (Pflicht-Kern für alle 6 Personen) | Zusatz A: Verschiedene Apnoe-Typen (`OSA, CSA, Hypopnoe`) | Zusatz B: Einbezug eines Schlafphasen-Modells (`W, N1, N2, N3, REM`) |
| :--- | :--- | :--- | :--- |
| **Fragestellung** | Liegt im 60s-Fenster eine **Apnoe (`1`) oder Keine Apnoe (`0`)** vor, und welche Sensorkombination (über mehrere Datensätze) funktioniert am besten? | Handelt es sich um **Normal (`0`), OSA (`1`), CSA (`2`) oder Hypopnoe (`3`)**? | Wie stark verbessert sich die Apnoe-Erkennung & AHI-Berechnung, wenn wir **Wachphasen filtern (`TST`)** und die **Schlafphase als Feature** übergeben? |
| **Benötigte Sensoren** | **Variabel per Toggle (`ACTIVE_CHANNELS`):** Schon ab `SpO2 + HR` oder `SpO2 + HR + THOR + ABDO` lauffähig; beliebig erweiterbar um `FLOW`, `EEG_C4`, `ECG`. | Zwingend **`FLOW` + `THOR` + `ABDO` + `SpO2`** (auf `10 Hz` statt `1 Hz` für Kurvenform & Paradoxie). | Mindestens **`1 EEG-Kanal (C4-A1)`** (oder ersatzweise `HRV + Bewegung` für grobe Schlaf-Wach-Trennung). |
| **Wichtigste Zusatz-Features** | 1-Hz-Zeitreihen bzw. 5 Basis-Statistiken (`Mean, Std, Min, Max, Peak-to-Peak`) pro aktivem Kanal. | **Kreuzkorrelation `corr(THOR, ABDO)`** (negativ bei OSA, flach bei CSA) + relativer `FLOW`-Drop (`>=90%` vs. `>=30%`). | Relative **EEG-Bandpower (`δ, θ, α, σ, β`)**, **Schlafstadien-Wahrscheinlichkeiten `P(W..REM)`** & **Wahre Schlafzeit `TST`**. |
| **Modell-Anpassung** | **Random Forest / LightGBM** + **Simples 1D-CNN** mit `Sigmoid`-Output & `BinaryCrossentropy`. | Output `Dense(4, 'softmax')` + **`class_weight='balanced'` / Focal Loss** gegen die Seltenheit von CSA & Hypopnoe. | Entweder **Post-Processing-Filter** (`if Stage == Wake: Apnea = 0` & `AHI = Events / TST`) oder **6 Zusatz-Spalten** im Feature-Vektor. |
| **Rolle im KES-Paper** | Bildet den **kompletten Hauptteil** (*Methods, Multi-Dataset Pipeline, Sensor-Toggle Ablation Tables & Plots*). | Entweder als **kleines Zusatz-Experiment in WP 4/5** ODER als detaillierter **Diskussionspunkt in „Future Work“**. | Entweder als **schnelles Zusatz-Experiment mit `YASA`/EEG-Features** ODER als zweiter **Hauptpunkt in „Discussion & Future Work“**. |
