<?php
// ============================================================================
// Teamprojekt AIN7 - Backend Storage API (PHP)
// Ermöglicht das Speichern & Laden aller Mindmap-Daten auf jedem PHP-Webserver.
// ============================================================================

header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Methods: GET, POST, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type, Authorization, X-Requested-With");
header("Content-Type: application/json; charset=UTF-8");
header("Cache-Control: no-store, no-cache, must-revalidate, max-age=0");

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit;
}

$dataDir = __DIR__ . '/data';
$dataFile = $dataDir . '/mindmap_data.json';

if (!file_exists($dataDir)) {
    @mkdir($dataDir, 0775, true);
}

$action = isset($_GET['action']) ? $_GET['action'] : '';

// 1. Health check & Ping
if ($action === 'ping') {
    echo json_encode([
        'status' => 'ok',
        'backend' => 'php',
        'dataFileExists' => file_exists($dataFile),
        'serverTime' => date('c')
    ]);
    exit;
}

// 2. Load data (GET)
if ($_SERVER['REQUEST_METHOD'] === 'GET' || $action === 'load') {
    if (file_exists($dataFile)) {
        $content = file_get_contents($dataFile);
        if ($content !== false && strlen(trim($content)) > 0) {
            echo $content;
            exit;
        }
    }
    echo json_encode([
        'status' => 'ok',
        'exists' => false,
        'message' => 'No saved data found yet on server.'
    ]);
    exit;
}

// 3. Save data (POST)
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $rawInput = file_get_contents('php://input');
    if (!$rawInput) {
        http_response_code(400);
        echo json_encode(['success' => false, 'error' => 'Empty request body']);
        exit;
    }

    $decoded = json_decode($rawInput, true);
    if ($decoded === null) {
        http_response_code(400);
        echo json_encode(['success' => false, 'error' => 'Invalid JSON']);
        exit;
    }

    $decoded['updatedAt'] = date('c');

    $encoded = json_encode($decoded, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
    
    // Save atomically with file lock
    $result = file_put_contents($dataFile, $encoded, LOCK_EX);

    if ($result !== false) {
        echo json_encode([
            'success' => true,
            'message' => 'Data saved successfully',
            'bytesWritten' => $result,
            'updatedAt' => $decoded['updatedAt']
        ]);
    } else {
        http_response_code(500);
        echo json_encode([
            'success' => false,
            'error' => 'Could not write to data file. Check directory permissions for /data.'
        ]);
    }
    exit;
}

http_response_code(405);
echo json_encode(['error' => 'Method not allowed']);
