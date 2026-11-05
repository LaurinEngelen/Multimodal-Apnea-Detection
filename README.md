# Multimodale Schlafapnoe-Detektion (HTWG Konstanz - AIN7)

Forschungsprojekt zur automatisierten Erkennung von obstruktiver, zentraler und gemischter Schlafapnoe mittels multimodaler Biosignale (SpO2, Atmung, EKG, EEG) und Machine Learning / Deep Learning im Wintersemester 2026/27.

---

## 🌐 Live Mindmap & Dokumenten-Zentrale (GitHub Pages)
Die interaktive Web-App kann direkt ohne Installation im Browser aufgerufen werden:  
👉 **[https://laurinengelen.github.io/Multimodal-Apnea-Detection/](https://laurinengelen.github.io/Multimodal-Apnea-Detection/)**

### Features der Web-App
- **🗺️ Interaktiver Wissensgraph:** 7 Hauptäste (`Sensoren`, `Medizin`, `Datensätze`, `Datenpipeline`, `Modelle`, `Paper`, `Evaluation`).
- **✏️ Voll editierbar:** Themen direkt im Browser anpassen, neue Punkte hinzufügen (`➕`) oder löschen (`🗑️`), automatische Speicherung im `localStorage` und JSON-Backup-Export (`⚙️`).
- **🌙 / ☀️ Dark Mode:** Vollständig integriertes helles und dunkles Farbschema.
- **📚 Integrierter Dokumenten-Viewer:**
  - Komplette 18-Spalten-Literaturmatrix (`.xlsx`) als interaktive Tabelle (inkl. CSV-Export).
  - Formatierter Preview-Modus für den Strategie- & Entscheidungsbericht sowie das Tiefenrecherche-Kompendium.
  - Vollständige Anzeige aller 6 Kern-Paper und Vorlesungs-PDFs.
- **📋 Aufgaben & Notizen:** To-Do-Board für offene Fragestellungen bis zum nächsten Betreuer-Meeting.

---

## 📁 Repository-Struktur

```text
Teamprojekt/
├── docs/                          # GitHub Pages Verzeichnis (Mindmap & In-Browser-Viewer)
│   ├── index.html                 # Haupteinstiegspunkt für GitHub Pages
│   ├── Papers/                    # PDF-Kopien für den In-Browser-Viewer
│   └── Zusammenfassungen/         # Berichte & Tabellen für den Viewer
│
├── Papers/                        # Originale wissenschaftliche Papersammlung
├── Zusammenfassungen/             # Originale Notizen, Markdown-Berichte & Excel-Matrix
│
├── src/                           # (Geplant) Python-Pipeline, Feature-Extraction, Modelle
├── notebooks/                     # (Geplant) Jupyter Notebooks für EDA & Visualisierung
├── data/                          # (Geplant / .gitignored) NSRR-Datensätze (SHHS, MESA)
├── README.md                      # Projektdokumentation
└── .gitignore                     # Git-Ignore für Python, venv, Checkpoints etc.
```

---

## 💻 Lokale Ausführung
Die Website kann jederzeit auch lokal offline genutzt werden:  
Einfach `docs/index.html` per Doppelklick in einem beliebigen Browser öffnen.
