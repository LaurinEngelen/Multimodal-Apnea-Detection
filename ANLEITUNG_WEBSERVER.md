# 🚀 Anleitung: Webserver-Bereitstellung mit permanenter Speicherung

Dieses Projekt ist so aufgebaut, dass du **einfach den gesamten Ordner `docs/` (oder das Projektverzeichnis) auf einen beliebigen Webserver kopieren kannst**. Alle Änderungen (Knoten bearbeiten, neue Punkte, Team-Zuweisungen, Positionen, Notizen/Aufgaben) werden automatisch dauerhaft auf dem Webserver in der Datei `data/mindmap_data.json` gespeichert.

---

## 📁 Was enthält der Ordner?

- **`index.html`** – Die vollständige Web-App (Mindmap, Dokumenten-Reader, Excel-Matrix, Aufgaben, Team-Manager).
- **`api.php`** – Backend für jeden normalen PHP-fähigen Webhosting-Server (Apache, Nginx, Plesk, cPanel, Uni-Server, XAMPP).
- **`server.py`** – Portabler, eigenständiger Python-3-Webserver (benötigt keine Installationen, reines Standard-Python).
- **`server.js`** – Portabler Node.js-Webserver (benötigt kein `npm install`, reine Standardmodule).
- **`data/mindmap_data.json`** – Die zentrale JSON-Datei, in der alle Veränderungen des Teams sicher gespeichert werden.
- **`Papers/` & `Zusammenfassungen/`** – Alle hinterlegten PDFs, Berichte und Tabellen.

---

## 🛠️ Wähle deine Bereitstellungsmethode:

### Option 1: Standard PHP / Webhosting (Empfohlen für Webspaces)
> Funktioniert auf jedem normalen Shared-Webhosting (Hetzner, Strato, IONOS, All-In-One, Uni-Webspace, Plesk, cPanel, XAMPP).

1. Kopiere den gesamten Ordner (z.B. per FTP / SCP oder Dateimanager) auf deinen Webserver in dein Web-Verzeichnis (z. B. `public_html/apnea/` oder `/var/www/html/`).
2. Stelle sicher, dass der Webserver Schreibrechte auf den Ordner `data/` hat (`chmod 775 data` falls nötig).
3. Öffne die URL im Browser: `https://dein-server.de/apnea/`
4. **Fertig!** Das Frontend erkennt `api.php` automatisch. In der Kopfzeile leuchtet `🟢 Server synchronisiert`. Jede Änderung wird sofort auf dem Server gespeichert.

---

### Option 2: Linux / VPS / Raspberry Pi mit Python
> Wenn du einen eigenen Linux-Server, VPS oder PC im LAN hast, auf dem Python 3 installiert ist.

1. Kopiere den Ordner auf den Server.
2. Wechsle in den Ordner:
   ```bash
   cd docs
   ```
3. Starte den Server:
   ```bash
   python3 server.py 8080
   ```
4. Öffne im Browser: `http://<server-ip>:8080`
5. *(Optional als Hintergrund-Dienst mit `nohup` oder `systemd`):*
   ```bash
   nohup python3 server.py 8080 > server.log 2>&1 &
   ```

---

### Option 3: Mit Node.js
> Wenn Node.js auf dem Server verfügbar ist:

```bash
cd docs
node server.js 8080
```

---

### Option 4: Mit Docker
> Für isolierte Container-Bereitstellung:

```bash
docker compose up -d
```
Die Anwendung ist sofort unter `http://localhost:8080` erreichbar. Die Daten bleiben im gemounteten Verzeichnis `./data` erhalten.

---

## 🔒 Wie funktioniert die Speicherung?

1. **Automatischer Server-Check:** Beim Laden prüft die App, ob ein Backend (`api.php` oder Python/Node-Route `/api/data`) antwortet.
2. **Synchronisation:**
   - Sobald eine Verbindung besteht, lädt die App den aktuellen Stand von `data/mindmap_data.json`.
   - Bei jeder Aktion (Knoten hinzufügen, Text ändern, Zuweisen von Teammitgliedern, Verschieben, Notizen abhaken) wird die Änderung automatisch im Hintergrund an den Server gesendet.
3. **Offline- & GitHub-Pages-Fallback:** Falls die Datei ohne Server geöffnet wird (z. B. als lokale Datei oder auf GitHub Pages), speichert die App nahtlos im Browser (`localStorage`), sodass nichts verloren geht.
4. **Backups:** Um ein vollständiges Backup des Datenstands zu erstellen, musst du lediglich die kleine Datei `data/mindmap_data.json` kopieren.
