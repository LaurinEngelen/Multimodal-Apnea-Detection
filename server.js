/**
 * ============================================================================
 * Teamprojekt AIN7 - Portabler Node.js Webserver
 * Startet einen Server ohne externe npm-Pakete (nur Node.js Standardmodule).
 * Speichert alle Änderungen in 'data/mindmap_data.json'.
 *
 * Starten mit:
 *     node server.js
 *     oder
 *     PORT=8080 node server.js
 * ============================================================================
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = parseInt(process.env.PORT || process.argv[2] || 8080, 10);
const BASE_DIR = __dirname;
const DATA_DIR = path.join(BASE_DIR, 'data');
const DATA_FILE = path.join(DATA_DIR, 'mindmap_data.json');

if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.pdf': 'application/pdf',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  '.md': 'text/markdown; charset=utf-8'
};

const server = http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0');

  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  const [reqPath, queryStr] = req.url.split('?');

  if (reqPath === '/api/data' || reqPath === '/api.php' || reqPath === '/api') {
    if (req.method === 'GET') {
      if (queryStr && queryStr.includes('action=ping')) {
        res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({
          status: 'ok',
          backend: 'node',
          dataFileExists: fs.existsSync(DATA_FILE),
          serverTime: new Date().toISOString()
        }));
        return;
      }

      if (fs.existsSync(DATA_FILE)) {
        const content = fs.readFileSync(DATA_FILE, 'utf-8');
        res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(content);
      } else {
        res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ status: 'ok', exists: false, message: 'No data yet' }));
      }
      return;
    }

    if (req.method === 'POST') {
      let body = '';
      req.on('data', chunk => { body += chunk; });
      req.on('end', () => {
        try {
          const parsed = JSON.parse(body);
          parsed.updatedAt = new Date().toISOString();
          fs.writeFileSync(DATA_FILE, JSON.stringify(parsed, null, 2), 'utf-8');
          res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
          res.end(JSON.stringify({ success: true, updatedAt: parsed.updatedAt }));
          console.log(`[${new Date().toLocaleTimeString()}] Saved updated mindmap data (${body.length} bytes)`);
        } catch (err) {
          res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
          res.end(JSON.stringify({ success: false, error: err.message }));
        }
      });
      return;
    }
  }

  // Static file serving
  let rawUrlPath = decodeURIComponent(reqPath === '/' ? '/index.html' : reqPath);
  let filePath = path.join(BASE_DIR, rawUrlPath);

  if (!fs.existsSync(filePath)) {
    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('404 Not Found');
    return;
  }

  const stat = fs.statSync(filePath);
  if (stat.isDirectory()) {
    filePath = path.join(filePath, 'index.html');
    if (!fs.existsSync(filePath)) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('404 Not Found');
      return;
    }
  }

  const ext = path.extname(filePath).toLowerCase();
  const contentType = MIME_TYPES[ext] || 'application/octet-stream';

  res.writeHead(200, { 'Content-Type': contentType });
  fs.createReadStream(filePath).pipe(res);
});

server.listen(PORT, () => {
  console.log('='.repeat(65));
  console.log(`  Teamprojekt Webserver läuft!`);
  console.log(`  URL: http://localhost:${PORT}`);
  console.log(`  Speicherort für Änderungen: ${DATA_FILE}`);
  console.log(`  Beenden mit Strg+C`);
  console.log('='.repeat(65));
});
