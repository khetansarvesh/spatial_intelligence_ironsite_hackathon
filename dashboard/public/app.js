/**
 * SiteIQ Dashboard Application
 */

const API_BASE = '/api';
let currentReport = null;
let activityChart = null;
let toolChart = null;

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    loadReports();
});

function initializeApp() {
    console.log('SiteIQ Dashboard initialized');
}

// ============================================================================
// Event Listeners
// ============================================================================

function setupEventListeners() {
    // Report selection
    document.getElementById('reportSelect').addEventListener('change', handleReportChange);

    // Upload buttons
    document.getElementById('uploadBtn').addEventListener('click', openUploadModal);
    document.getElementById('uploadBtnEmpty').addEventListener('click', openUploadModal);

    // Modal
    document.querySelector('.modal-close').addEventListener('click', closeUploadModal);
    document.querySelector('.modal').addEventListener('click', (e) => {
        if (e.target.classList.contains('modal')) {
            closeUploadModal();
        }
    });

    // File upload
    const uploadArea = document.getElementById('uploadArea');
    const videoInput = document.getElementById('videoInput');

    uploadArea.addEventListener('click', () => videoInput.click());
    videoInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // Chat
    const chatInput = document.getElementById('chatInput');
    const chatSend = document.getElementById('chatSend');

    chatSend.addEventListener('click', sendChatMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !chatSend.disabled) {
            sendChatMessage();
        }
    });
}

// ============================================================================
// API Functions
// ============================================================================

async function api(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, options);
        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

async function loadReports() {
    try {
        const data = await api('/reports');
        const select = document.getElementById('reportSelect');

        select.innerHTML = '<option value="">Select a report...</option>';

        data.reports.forEach(report => {
            const option = document.createElement('option');
            option.value = report.name;
            option.textContent = report.name;
            select.appendChild(option);
        });

        // Auto-select first report if available
        if (data.reports.length > 0) {
            select.value = data.reports[0].name;
            handleReportChange();
        }
    } catch (error) {
        console.error('Failed to load reports:', error);
    }
}

async function loadReport(filename) {
    showLoading(true);

    try {
        const report = await api(`/report/${filename}`);
        currentReport = report;
        displayReport(report);
    } catch (error) {
        console.error('Failed to load report:', error);
        alert('Failed to load report');
    } finally {
        showLoading(false);
    }
}

async function queryAgent(question) {
    const reportFile = document.getElementById('reportSelect').value;
    if (!reportFile) return;

    try {
        const data = await api('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                report_file: reportFile,
                question: question,
                provider: 'openai' // Can be made configurable
            })
        });

        return data.answer;
    } catch (error) {
        console.error('Agent query failed:', error);
        throw error;
    }
}

// ============================================================================
// Display Functions
// ============================================================================

function displayReport(report) {
    // Show dashboard, hide empty state
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('dashboardContent').style.display = 'block';

    // Update overview cards
    updateOverviewCards(report);

    // Update charts
    updateCharts(report);

    // Update insights
    updateInsights(report);

    // Enable chat
    document.getElementById('chatInput').disabled = false;
    document.getElementById('chatSend').disabled = false;

    // Clear previous chat
    document.getElementById('chatMessages').innerHTML = '';
}

function updateOverviewCards(report) {
    // Productivity score
    const score = (report.productivity_score * 100).toFixed(1);
    document.getElementById('productivityScore').textContent = `${score}%`;
    document.getElementById('productivityRating').textContent = getProductivityRating(report.productivity_score);

    // Session duration
    document.getElementById('sessionDuration').textContent = formatTime(report.session_duration);

    // Idle time
    document.getElementById('idleTime').textContent = formatTime(report.idle_time);
    document.getElementById('idlePercentage').textContent = `${report.idle_percentage.toFixed(1)}% of session`;

    // Most used tool
    document.getElementById('mostUsedTool').textContent = report.most_used_tool || 'None';
    document.getElementById('toolSwitches').textContent = `${report.tool_switches} tool switches`;
}

function updateCharts(report) {
    // Activity breakdown chart
    const activityData = Object.entries(report.activity_breakdown).map(([name, data]) => ({
        name,
        time: data.total_time,
        percentage: data.percentage
    }));

    // Sort by time
    activityData.sort((a, b) => b.time - a.time);

    const activityLabels = activityData.map(d => formatActivityName(d.name));
    const activityValues = activityData.map(d => d.percentage);
    const activityColors = activityData.map(d => getActivityColor(d.name));

    if (activityChart) {
        activityChart.destroy();
    }

    const activityCtx = document.getElementById('activityChart').getContext('2d');
    activityChart = new Chart(activityCtx, {
        type: 'doughnut',
        data: {
            labels: activityLabels,
            datasets: [{
                data: activityValues,
                backgroundColor: activityColors,
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            return `${label}: ${value.toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });

    // Tool usage chart
    const toolData = Object.entries(report.tool_usage || {}).map(([name, data]) => ({
        name,
        time: data.total_time,
        count: data.usage_count
    }));

    toolData.sort((a, b) => b.time - a.time);

    const toolLabels = toolData.map(d => d.name);
    const toolTimes = toolData.map(d => d.time / 60); // Convert to minutes

    if (toolChart) {
        toolChart.destroy();
    }

    const toolCtx = document.getElementById('toolChart').getContext('2d');
    toolChart = new Chart(toolCtx, {
        type: 'bar',
        data: {
            labels: toolLabels,
            datasets: [{
                label: 'Usage Time (minutes)',
                data: toolTimes,
                backgroundColor: '#2563eb',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: (value) => `${value.toFixed(1)}m`
                    }
                }
            }
        }
    });
}

function updateInsights(report) {
    const container = document.getElementById('insightsList');
    container.innerHTML = '';

    // Insights
    if (report.insights && report.insights.length > 0) {
        const insightsTitle = document.createElement('h4');
        insightsTitle.textContent = 'Insights';
        insightsTitle.style.fontSize = '0.875rem';
        insightsTitle.style.fontWeight = '600';
        insightsTitle.style.marginBottom = '0.5rem';
        container.appendChild(insightsTitle);

        report.insights.forEach(insight => {
            const item = document.createElement('div');
            item.className = 'insight-item';
            item.innerHTML = `<strong>💡</strong> ${insight}`;
            container.appendChild(item);
        });
    }

    // Recommendations
    if (report.recommendations && report.recommendations.length > 0) {
        const recTitle = document.createElement('h4');
        recTitle.textContent = 'Recommendations';
        recTitle.style.fontSize = '0.875rem';
        recTitle.style.fontWeight = '600';
        recTitle.style.margin = '1rem 0 0.5rem';
        container.appendChild(recTitle);

        report.recommendations.forEach(rec => {
            const item = document.createElement('div');
            item.className = 'recommendation-item';
            item.innerHTML = `<strong>→</strong> ${rec}`;
            container.appendChild(item);
        });
    }
}

// ============================================================================
// Chat Functions
// ============================================================================

async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const question = input.value.trim();

    if (!question) return;

    // Add user message
    addChatMessage('user', question);
    input.value = '';

    // Show loading
    const loadingId = addChatMessage('loading', 'Thinking...');

    try {
        const answer = await queryAgent(question);
        removeChatMessage(loadingId);
        addChatMessage('assistant', answer);
    } catch (error) {
        removeChatMessage(loadingId);
        addChatMessage('assistant', 'Sorry, I encountered an error. Please try again.');
    }
}

function addChatMessage(type, text) {
    const messages = document.getElementById('chatMessages');
    const message = document.createElement('div');
    const id = `msg-${Date.now()}`;

    message.id = id;
    message.className = `chat-message ${type}`;
    message.textContent = text;

    messages.appendChild(message);
    messages.scrollTop = messages.scrollHeight;

    return id;
}

function removeChatMessage(id) {
    const message = document.getElementById(id);
    if (message) {
        message.remove();
    }
}

// ============================================================================
// Upload Functions
// ============================================================================

function openUploadModal() {
    document.getElementById('uploadModal').classList.add('active');
}

function closeUploadModal() {
    document.getElementById('uploadModal').classList.remove('active');
    document.getElementById('uploadProgress').style.display = 'none';
    document.getElementById('uploadArea').style.display = 'block';
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFileUpload(file);
    }
}

async function handleFileUpload(file) {
    if (!file.type.startsWith('video/')) {
        alert('Please select a video file');
        return;
    }

    // Show progress
    document.getElementById('uploadArea').style.display = 'none';
    document.getElementById('uploadProgress').style.display = 'block';

    const formData = new FormData();
    formData.append('video', file);

    const quickProcessing = document.getElementById('quickProcessing').checked;
    if (quickProcessing) {
        formData.append('max_frames', '300');
    }

    try {
        updateProgress(10, 'Uploading video...');

        const response = await fetch(`${API_BASE}/process-video`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        updateProgress(50, 'Processing video...');

        // Simulate processing progress (actual processing happens on server)
        await simulateProgress();

        const result = await response.json();

        updateProgress(100, 'Complete!');

        // Reload reports and select the new one
        await loadReports();
        document.getElementById('reportSelect').value = result.report_file;
        await loadReport(result.report_file);

        setTimeout(() => {
            closeUploadModal();
        }, 1000);

    } catch (error) {
        console.error('Upload failed:', error);
        alert('Failed to process video. Please try again.');
        closeUploadModal();
    }
}

function updateProgress(percent, text) {
    document.getElementById('progressFill').style.width = `${percent}%`;
    document.getElementById('progressText').textContent = text;
}

async function simulateProgress() {
    for (let i = 50; i < 90; i += 5) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        updateProgress(i, 'Processing video...');
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

function handleReportChange() {
    const filename = document.getElementById('reportSelect').value;
    if (filename) {
        loadReport(filename);
    }
}

function showLoading(show) {
    document.getElementById('loadingSpinner').style.display = show ? 'flex' : 'none';
}

function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}

function formatActivityName(name) {
    return name
        .split('_')
        .map(word => word.charAt(0) + word.slice(1).toLowerCase())
        .join(' ');
}

function getActivityColor(name) {
    const colors = {
        'ACTIVE_TOOL_USE': '#10b981',
        'PRECISION_WORK': '#3b82f6',
        'MATERIAL_HANDLING': '#8b5cf6',
        'SETUP_CLEANUP': '#f59e0b',
        'SEARCHING': '#ef4444',
        'TRAVELING': '#64748b',
        'IDLE': '#cbd5e1'
    };
    return colors[name] || '#94a3b8';
}

function getProductivityRating(score) {
    if (score >= 0.8) return 'Excellent ⭐⭐⭐';
    if (score >= 0.7) return 'Good ⭐⭐';
    if (score >= 0.6) return 'Fair ⭐';
    if (score >= 0.5) return 'Below Average';
    return 'Poor';
}
