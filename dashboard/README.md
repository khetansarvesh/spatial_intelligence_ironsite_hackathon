# SiteIQ Dashboard - Node.js Web Interface

Modern web dashboard for SiteIQ productivity analytics built with Node.js/Express and vanilla JavaScript.

## Features

- 📊 **Real-time Metrics** - Productivity score, session duration, idle time, tool usage
- 📈 **Interactive Charts** - Activity breakdown (pie chart) and tool usage (bar chart)
- 💡 **AI Insights** - Automated recommendations and productivity insights
- 🤖 **LLM Agent Chat** - Natural language queries about productivity
- 📤 **Video Upload** - Process new videos directly from the browser
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile

## Quick Start

### 1. Install Dependencies

```bash
cd dashboard
npm install
```

### 2. Start the Server

```bash
npm start
```

The dashboard will be available at: **http://localhost:3000**

### 3. For Development (with auto-reload)

```bash
npm run dev
```

## Prerequisites

- **Node.js** 16+ (with npm)
- **Python 3.12+** with SiteIQ dependencies installed
- **OpenAI or Anthropic API key** (for LLM agent features)

Set your API key:
```bash
export OPENAI_API_KEY=your_key
# OR
export ANTHROPIC_API_KEY=your_key
```

## Usage

### View Existing Reports

1. Open http://localhost:3000
2. Select a report from the dropdown
3. View metrics, charts, and insights
4. Chat with the AI agent about productivity

### Process New Video

1. Click "Upload Video" button
2. Drag & drop or select a video file
3. Optionally enable "Quick processing" (first 300 frames only)
4. Wait for processing to complete
5. View the generated report

### Query the Agent

In the "Ask SiteIQ" panel:
- Type questions like "What tools were used most?"
- Get data-driven answers in natural language
- Examples:
  - "What was the overall productivity score?"
  - "How much time was spent idle?"
  - "How can we improve productivity?"

## API Endpoints

The server exposes these REST API endpoints:

### Reports
- `GET /api/reports` - List all available reports
- `GET /api/report/:filename` - Get report data
- `GET /api/report/:filename/summary` - Get text summary

### Agent
- `POST /api/query` - Query the LLM agent
  ```json
  {
    "report_file": "report.json",
    "question": "What tools were used?",
    "provider": "openai"
  }
  ```

### Video Processing
- `POST /api/process-video` - Upload and process video
  - Multipart form with video file
  - Optional: `max_frames` parameter

### Health
- `GET /api/health` - Server health check

## Project Structure

```
dashboard/
├── server.js           # Express server
├── package.json        # Node dependencies
├── public/             # Static files
│   ├── index.html      # Main page
│   ├── app.js          # Frontend logic
│   └── styles.css      # Styles
└── README.md           # This file
```

## Configuration

Environment variables (create `.env` file):

```bash
PORT=3000                      # Server port
OPENAI_API_KEY=your_key       # For OpenAI agent
ANTHROPIC_API_KEY=your_key    # For Anthropic agent
```

## Technology Stack

**Backend:**
- Express.js - Web framework
- Multer - File upload handling
- CORS - Cross-origin support
- Child Process - Python integration

**Frontend:**
- Vanilla JavaScript (ES6+)
- Chart.js - Data visualization
- CSS3 - Modern styling
- Fetch API - HTTP requests

## Development

### Adding New Features

1. **New API Endpoint**: Edit `server.js`
   ```javascript
   app.get('/api/your-endpoint', async (req, res) => {
     // Your logic here
   });
   ```

2. **Frontend Function**: Edit `public/app.js`
   ```javascript
   async function yourFunction() {
     const data = await api('/your-endpoint');
     // Handle data
   }
   ```

3. **UI Component**: Edit `public/index.html` and `public/styles.css`

### Common Customizations

**Change Chart Colors:**
Edit the `getActivityColor()` function in `app.js`

**Modify Agent Provider:**
Change `provider: 'openai'` to `provider: 'anthropic'` in the `queryAgent()` function

**Adjust Upload Limits:**
Modify `multer` configuration in `server.js`

## Troubleshooting

### Port Already in Use
```bash
# Change port in .env or:
PORT=3001 npm start
```

### Python Script Errors
Check that:
- Python 3.12+ is installed
- SiteIQ dependencies are installed (`pip install -r requirements.txt`)
- Python scripts are in the parent directory

### API Key Issues
Verify environment variables:
```bash
echo $OPENAI_API_KEY
# or
echo $ANTHROPIC_API_KEY
```

### CORS Errors
The server includes CORS middleware. If issues persist, check browser console.

## Performance Tips

1. **Quick Processing**: Use the "Quick processing" checkbox for faster results (first 300 frames)
2. **Frame Skipping**: Process every Nth frame by modifying `main.py` call in server
3. **Caching**: Consider adding Redis for report caching
4. **Load Balancing**: Use PM2 for production deployments

## Production Deployment

### Using PM2

```bash
# Install PM2
npm install -g pm2

# Start
pm2 start server.js --name siteiq-dashboard

# Monitor
pm2 monit

# Auto-restart on reboot
pm2 startup
pm2 save
```

### Using Docker

```bash
# Build
docker build -t siteiq-dashboard .

# Run
docker run -p 3000:3000 \
  -e OPENAI_API_KEY=your_key \
  -v $(pwd)/../:/app/project \
  siteiq-dashboard
```

### Environment Variables for Production

```bash
NODE_ENV=production
PORT=3000
OPENAI_API_KEY=your_production_key
```

## Security Notes

⚠️ **Important for Production:**

1. Add authentication middleware
2. Implement rate limiting
3. Validate file uploads (type, size, malware scan)
4. Use HTTPS
5. Sanitize user inputs
6. Add CSRF protection
7. Keep API keys secure (never commit to git)

## Examples

### Integrate with Custom Workflow

```javascript
// Custom script to process video
const FormData = require('form-data');
const fs = require('fs');
const fetch = require('node-fetch');

const form = new FormData();
form.append('video', fs.createReadStream('video.mp4'));

const response = await fetch('http://localhost:3000/api/process-video', {
  method: 'POST',
  body: form
});

const result = await response.json();
console.log('Report:', result.report_file);
```

### Scheduled Processing

```bash
# Cron job to process videos daily
0 2 * * * node /path/to/process-daily-videos.js
```

## License

MIT

## Support

For issues and questions:
1. Check the main SiteIQ documentation
2. Review API logs in the terminal
3. Check browser console for frontend errors
4. Verify Python dependencies are installed
