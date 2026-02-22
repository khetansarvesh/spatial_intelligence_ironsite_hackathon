/**
 * SiteIQ Dashboard Server
 * Express.js backend for productivity analytics dashboard
 */

const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs').promises;
const { spawn } = require('child_process');
const multer = require('multer');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// File upload configuration
const upload = multer({
  dest: '../uploads/',
  limits: { fileSize: 500 * 1024 * 1024 } // 500MB limit
});

// Path to Python scripts
const PYTHON_PATH = 'python3';
const PROJECT_ROOT = path.join(__dirname, '..');

/**
 * Execute Python script and return result
 */
function executePython(scriptPath, args = []) {
  return new Promise((resolve, reject) => {
    console.log(`Executing: ${PYTHON_PATH} ${scriptPath} ${args.join(' ')}`);
    console.log(`Working directory: ${PROJECT_ROOT}`);

    const python = spawn(PYTHON_PATH, [scriptPath, ...args], {
      cwd: PROJECT_ROOT
    });

    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    python.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    python.on('error', (error) => {
      console.error('Python spawn error:', error);
      reject(new Error(`Failed to start Python: ${error.message}`));
    });

    python.on('close', (code) => {
      console.log(`Python exited with code ${code}`);
      if (stdout) console.log('stdout:', stdout.substring(0, 500));
      if (stderr) console.error('stderr:', stderr.substring(0, 500));

      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(stderr || `Python script failed with code ${code}`));
      }
    });
  });
}

// ============================================================================
// API Routes
// ============================================================================

/**
 * GET /api/reports
 * List all available reports
 */
app.get('/api/reports', async (req, res) => {
  try {
    const files = await fs.readdir(PROJECT_ROOT);
    const reports = files
      .filter(f => f.endsWith('_report.json'))
      .map(f => ({
        name: f,
        path: f,
        timestamp: f.replace('_report.json', '')
      }));

    res.json({ reports });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/report/:filename
 * Get specific report data
 */
app.get('/api/report/:filename', async (req, res) => {
  try {
    const reportPath = path.join(PROJECT_ROOT, req.params.filename);
    const data = await fs.readFile(reportPath, 'utf-8');
    const report = JSON.parse(data);

    res.json(report);
  } catch (error) {
    res.status(404).json({ error: 'Report not found' });
  }
});

/**
 * GET /api/report/:filename/summary
 * Get text summary of report
 */
app.get('/api/report/:filename/summary', async (req, res) => {
  try {
    const output = await executePython('query_agent.py', [
      '--report', req.params.filename,
      '--summary'
    ]);

    res.json({ summary: output });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/query
 * Query the LLM agent
 */
app.post('/api/query', async (req, res) => {
  const { report_file, question, provider = 'openai' } = req.body;

  if (!report_file || !question) {
    return res.status(400).json({ error: 'Missing report_file or question' });
  }

  try {
    const output = await executePython('query_agent.py', [
      '--report', report_file,
      '--provider', provider,
      question
    ]);

    // Parse the output to extract the answer
    // Format: "Q: question\n\nA: answer"
    const answerMatch = output.match(/A: ([\s\S]+)/);
    const answer = answerMatch ? answerMatch[1].trim() : output;

    res.json({ answer });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/process-video
 * Process a new video file
 */
app.post('/api/process-video', upload.single('video'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No video file uploaded' });
  }

  const videoPath = req.file.path;
  const outputName = req.body.output_name || `${Date.now()}_report.json`;
  const maxFrames = req.body.max_frames || null;

  try {
    // Build arguments
    const args = [
      'main.py',
      '--video', videoPath,
      '--output', outputName
    ];

    if (maxFrames) {
      args.push('--max-frames', maxFrames);
    }

    // Process video
    const output = await executePython(args[0], args.slice(1));

    // Clean up uploaded video
    await fs.unlink(videoPath);

    res.json({
      success: true,
      report_file: outputName,
      message: 'Video processed successfully'
    });
  } catch (error) {
    // Clean up on error
    try {
      await fs.unlink(videoPath);
    } catch {}

    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/summary
 * Get markdown summary from the pre-generated file
 */
app.get('/api/summary', async (req, res) => {
  try {
    const summaryPath = path.join(PROJECT_ROOT, 'outputs', 'productivity_summary.md');
    const markdown = await fs.readFile(summaryPath, 'utf-8');
    res.json({ markdown });
  } catch (error) {
    console.error('Summary error:', error);
    res.status(500).json({ error: 'Summary file not found' });
  }
});

/**
 * GET /api/video/annotated
 * Serve the annotated video file
 */
app.get('/api/video/annotated', async (req, res) => {
  const videoPath = path.join(PROJECT_ROOT, 'outputs', 'annotated_video.mp4');

  try {
    await fs.access(videoPath);
    res.sendFile(videoPath);
  } catch {
    res.status(404).json({ error: 'Annotated video not found' });
  }
});

/**
 * POST /api/ask
 * Ask a question using the CodeAct agent
 */
app.post('/api/ask', async (req, res) => {
  const { question } = req.body;

  if (!question) {
    return res.status(400).json({ error: 'Missing question' });
  }

  try {
    // Escape single quotes in question
    const escapedQuestion = question.replace(/'/g, "\\'").replace(/"/g, '\\"');

    const pythonCode = `
import sys
import json
sys.path.insert(0, '.')
from src.agent.agent import ask_with_code
result = ask_with_code('${escapedQuestion}')
print(json.dumps(result))
`;
    const output = await executePython('-c', [pythonCode]);
    const result = JSON.parse(output.trim());
    res.json({ answer: result.answer, code: result.code });
  } catch (error) {
    console.error('Ask error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/evidence
 * Get evidence timestamps for a question/answer pair
 */
app.post('/api/evidence', async (req, res) => {
  const { question, answer } = req.body;

  if (!question || !answer) {
    return res.status(400).json({ error: 'Missing question or answer' });
  }

  try {
    // Escape quotes
    const escapedQuestion = question.replace(/'/g, "\\'").replace(/"/g, '\\"');
    const escapedAnswer = answer.replace(/'/g, "\\'").replace(/"/g, '\\"');

    const pythonCode = `
import sys
import json
sys.path.insert(0, '.')
from src.agent.evidence import get_evidence
evidence = get_evidence('''${escapedQuestion}''', '''${escapedAnswer}''')
print(json.dumps(evidence))
`;
    const output = await executePython('-c', [pythonCode]);
    const evidence = JSON.parse(output.trim());
    res.json({ evidence });
  } catch (error) {
    console.error('Evidence error:', error);
    res.status(500).json({ error: error.message, evidence: [] });
  }
});

/**
 * GET /api/video/clip
 * Get a clipped portion of the annotated video
 * Query params: start (seconds), duration (seconds, default 2)
 */
app.get('/api/video/clip', async (req, res) => {
  const start = parseFloat(req.query.start) || 0;
  const duration = parseFloat(req.query.duration) || 2;

  const inputPath = path.join(PROJECT_ROOT, 'outputs', 'annotated_video.mp4');
  const outputPath = path.join(PROJECT_ROOT, 'outputs', `clip_${start}_${duration}.mp4`);

  try {
    // Use Python/OpenCV to clip the video
    const clipStart = Math.max(0, start - 1); // 1 sec before
    const clipDuration = duration + 2; // Add buffer

    const pythonCode = `
import cv2
import sys

input_path = '${inputPath}'
output_path = '${outputPath}'
start_time = ${clipStart}
duration = ${clipDuration}

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

start_frame = int(start_time * fps)
end_frame = min(int((start_time + duration) * fps), total_frames)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for i in range(end_frame - start_frame):
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()
out.release()
print('OK')
`;

    await executePython('-c', [pythonCode]);

    // Send the clipped video
    res.sendFile(outputPath, (err) => {
      // Clean up clip file after sending
      fs.unlink(outputPath).catch(() => {});
    });

  } catch (error) {
    console.error('Clip error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/health
 * Health check
 */
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    python: PYTHON_PATH,
    port: PORT
  });
});

// ============================================================================
// Start Server
// ============================================================================

app.listen(PORT, () => {
  console.log('╔════════════════════════════════════════════════════════╗');
  console.log('║        SiteIQ Dashboard Server                         ║');
  console.log('╚════════════════════════════════════════════════════════╝');
  console.log('');
  console.log(`  Server running at: http://localhost:${PORT}`);
  console.log(`  API endpoints:     http://localhost:${PORT}/api/`);
  console.log('');
  console.log('  Available routes:');
  console.log('    GET  /api/reports              - List all reports');
  console.log('    GET  /api/report/:filename     - Get report data');
  console.log('    GET  /api/summary              - Get markdown summary');
  console.log('    GET  /api/video/annotated      - Serve annotated video');
  console.log('    POST /api/ask                  - Ask question (CodeAct agent)');
  console.log('    POST /api/evidence             - Get evidence timestamps');
  console.log('    POST /api/process-video        - Process new video');
  console.log('    GET  /api/health               - Health check');
  console.log('');
  console.log('  Press Ctrl+C to stop');
  console.log('');
});
