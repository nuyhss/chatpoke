"""
Chat UI with model selector and chat history.
Access at: http://localhost:8000/ui
"""

import logging
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

logger = logging.getLogger("tilon.ui")
router = APIRouter(tags=["Chat UI"])
UI_INDEX_PATH = Path(__file__).resolve().parents[2] / "static" / "index.html"

CHAT_UI_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tilon AI Chatbot</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap');

        :root{
            --bg-0:#070a10;
            --bg-1:#0d1420;
            --panel:#111c2a;
            --panel-soft:#162336;
            --line:#27384f;
            --line-strong:#3a5878;
            --text:#edf4ff;
            --text-dim:#9cb2cc;
            --brand:#1aa39a;
            --brand-2:#f59e0b;
            --danger:#f87171;
            --ok:#22c55e;
            --glow-a:rgba(26,163,154,.24);
            --glow-b:rgba(245,158,11,.18);
        }

        *{margin:0;padding:0;box-sizing:border-box}
        html{height:100%;width:100%;overflow:hidden}

        body{
            font-family:'IBM Plex Sans KR','Noto Sans KR',sans-serif;
            color:var(--text);
            height:100dvh;
            min-height:100vh;
            width:100%;
            display:flex;
            background:
                radial-gradient(1200px 700px at 8% -12%, var(--glow-a), transparent 70%),
                radial-gradient(900px 600px at 95% 0%, var(--glow-b), transparent 60%),
                linear-gradient(155deg, var(--bg-0), var(--bg-1));
            overflow:hidden;
            overflow-x:hidden;
        }

        body.theme-rose{
            --bg-0:#180914;
            --bg-1:#27112a;
            --panel:#2e1834;
            --panel-soft:#3b2142;
            --line:#64406f;
            --line-strong:#8d5a95;
            --text:#fff1fa;
            --text-dim:#e2bdd8;
            --brand:#f472b6;
            --brand-2:#fb7185;
            --danger:#fb7185;
            --ok:#34d399;
            --glow-a:rgba(244,114,182,.30);
            --glow-b:rgba(251,113,133,.22);
        }

        body::before{
            content:"";
            position:fixed;
            inset:0;
            background-image:linear-gradient(rgba(255,255,255,.028) 1px, transparent 1px),linear-gradient(90deg,rgba(255,255,255,.02) 1px, transparent 1px);
            background-size:44px 44px;
            opacity:.35;
            pointer-events:none;
        }

        /* ── Sidebar (Chat History) ── */
        .sidebar{
            width:282px;
            background:linear-gradient(180deg, rgba(17,28,42,.95), rgba(12,20,32,.94));
            border-right:1px solid var(--line);
            backdrop-filter:blur(8px);
            display:flex;
            flex-direction:column;
            flex-shrink:0;
        }
        .sidebar-top{padding:14px}
        .new-chat-btn{
            width:100%;
            padding:11px;
            border-radius:12px;
            border:1px solid var(--line-strong);
            background:linear-gradient(120deg, rgba(26,163,154,.18), rgba(245,158,11,.12));
            color:var(--text);
            font-size:.86rem;
            font-weight:600;
            letter-spacing:.01em;
            cursor:pointer;
            display:flex;
            align-items:center;
            justify-content:center;
            gap:6px;
            transition:.2s ease;
        }
        .new-chat-btn:hover{transform:translateY(-1px);border-color:var(--brand);box-shadow:0 8px 20px rgba(16,142,135,.24)}
        .history-list{flex:1;overflow-y:auto;padding:6px 10px}
        .history-label{padding:0 14px 6px;font-size:.68rem;color:#7fa0c3;letter-spacing:.08em;text-transform:uppercase}
        .shelf-wrap{padding:0 10px 10px}
        .shelf-head{display:flex;align-items:center;justify-content:space-between;padding:0 4px 8px}
        .shelf-upload-btn{padding:5px 8px;border-radius:8px;border:1px solid var(--line-strong);background:rgba(26,163,154,.12);color:var(--text);font-size:.68rem;cursor:pointer}
        .shelf-upload-btn:hover{background:rgba(26,163,154,.22);border-color:var(--brand)}
        .shelf-list{max-height:190px;overflow-y:auto;padding:4px;border:1px solid var(--line);border-radius:10px;background:rgba(12,20,32,.65)}
        .shelf-empty{font-size:.72rem;color:#6f88a5;padding:10px 8px;text-align:center}
        .shelf-item{padding:7px 8px;border-radius:8px;border:1px solid transparent;display:flex;flex-direction:column;gap:3px;margin-bottom:4px;background:rgba(19,31,47,.55)}
        .shelf-item:hover{border-color:var(--line-strong);background:rgba(25,40,61,.72)}
        .shelf-item-name{background:none;border:none;color:#d7e8fb;font-size:.75rem;text-align:left;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:0}
        .shelf-item-name.active{color:#9ef7ed;font-weight:700}
        .shelf-item-meta{font-size:.66rem;color:#85a4c5}
        .history-item{
            padding:10px 12px;
            border-radius:10px;
            font-size:.8rem;
            color:var(--text-dim);
            border:1px solid transparent;
            cursor:pointer;
            margin-bottom:4px;
            display:flex;
            justify-content:space-between;
            align-items:center;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
            transition:.16s ease;
        }
        .history-item:hover{background:rgba(23,38,57,.74);border-color:var(--line)}
        .history-item.active{background:rgba(26,163,154,.14);border-color:rgba(26,163,154,.6);color:#f2fffd}
        .history-item .del{display:none;background:none;border:none;color:#7f96b2;cursor:pointer;font-size:.9rem;padding:0 2px;flex-shrink:0}
        .history-item:hover .del{display:block}
        .history-item .del:hover{color:var(--danger)}
        .sidebar-footer{padding:11px 12px;border-top:1px solid var(--line);font-size:.7rem;color:#6f88a5;text-align:center;letter-spacing:.04em}

        /* ── Main ── */
        .main{flex:1;display:flex;flex-direction:column;min-width:0;width:100%;overflow:hidden;animation:fadeIn .35s ease}

        /* Topbar */
        .topbar{
            padding:10px 20px;
            border-bottom:1px solid var(--line);
            display:flex;
            align-items:center;
            justify-content:space-between;
            flex-wrap:wrap;
            background:linear-gradient(180deg, rgba(15,24,37,.95), rgba(15,24,37,.8));
            flex-shrink:0;
            gap:12px;
            backdrop-filter:blur(8px);
        }
        .topbar h1{
            font-family:'Space Grotesk','IBM Plex Sans KR',sans-serif;
            font-size:1.02rem;
            color:#f3fbff;
            white-space:nowrap;
            letter-spacing:.04em;
            text-transform:uppercase;
        }
        .topbar-center{display:flex;align-items:center;gap:10px;min-width:0;flex:1;justify-content:center}
        .model-select{padding:6px 11px;border-radius:8px;border:1px solid var(--line-strong);background:#18283c;color:var(--text);font-size:.78rem;outline:none;cursor:pointer;max-width:100%}
        .theme-select{padding:5px 9px;border-radius:8px;border:1px solid var(--line-strong);background:#18283c;color:var(--text);font-size:.72rem;outline:none;cursor:pointer}
        .control-stack{display:flex;flex-direction:column;gap:4px;align-items:stretch}
        .web-toggle{display:flex;align-items:center;gap:7px;color:var(--text);font-size:.74rem;line-height:1.2;padding:4px 8px;border-radius:999px;border:1px solid var(--line-strong);background:rgba(26,163,154,.12);white-space:nowrap;user-select:none}
        .web-toggle input{width:15px;height:15px;accent-color:var(--brand)}

        .model-select:focus{border-color:var(--brand);box-shadow:0 0 0 3px rgba(26,163,154,.2)}
        .theme-select:focus{border-color:var(--brand);box-shadow:0 0 0 3px rgba(26,163,154,.2)}
        .topbar-right{display:flex;align-items:flex-start;gap:10px;font-size:.72rem;flex-wrap:wrap;justify-content:flex-end}
        .status-badge{display:flex;align-items:center;gap:4px;color:#8ea4bf;background:rgba(18,33,51,.65);border:1px solid var(--line);padding:3px 8px;border-radius:999px}
        .dot{width:6px;height:6px;border-radius:50%}.dot.green{background:var(--ok)}.dot.red{background:#ef4444}
        .topbar-btn{padding:5px 9px;border-radius:7px;border:1px solid var(--line-strong);background:transparent;color:#c0d3e8;font-size:.7rem;cursor:pointer;transition:.16s ease}
        .topbar-btn:hover{background:rgba(26,163,154,.16);color:#fff;border-color:var(--brand)}

        /* Messages */
        .messages{flex:1;min-height:0;overflow-y:auto;padding:22px 16px;scroll-behavior:smooth}
        .msg-wrap{max-width:840px;margin:0 auto 16px;animation:rise .26s ease both}
        .message{display:flex;gap:10px}
        .message .avatar{width:30px;height:30px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:.75rem;flex-shrink:0;font-weight:700}
        .message.user .avatar{background:#13343d;color:#93f3e7;border:1px solid rgba(26,163,154,.55)}
        .message.assistant .avatar{background:#3a2407;color:#ffd39a;border:1px solid rgba(245,158,11,.5)}
        .message.system .avatar{background:#2a1725;color:#f7b4dd;font-size:.65rem;border:1px solid #834c73}
        .message .body{flex:1;min-width:0}
        .message .sender{font-size:.71rem;color:#8098b4;margin-bottom:3px;font-weight:600;display:flex;align-items:center;gap:6px}
        .message .content{font-size:.87rem;line-height:1.65;color:#e3edf8;white-space:pre-wrap;word-break:break-word}
        .message.assistant .content{background:linear-gradient(180deg, rgba(23,36,55,.93), rgba(17,28,43,.9));padding:13px 16px;border-radius:14px;border:1px solid var(--line);box-shadow:0 8px 24px rgba(0,0,0,.2)}
        .mode-tag{font-size:.58rem;padding:1px 5px;background:rgba(26,163,154,.18);color:#8df5e8;border-radius:4px;border:1px solid rgba(26,163,154,.35)}
        .sources{margin-top:6px;display:flex;flex-wrap:wrap;gap:5px}
        .source-tag{font-size:.66rem;padding:2px 8px;background:rgba(245,158,11,.14);color:#ffd9a0;border-radius:999px;border:1px solid rgba(245,158,11,.4)}
        .file-badge{font-size:.72rem;padding:4px 9px;background:rgba(26,163,154,.15);color:#9ef7ed;border-radius:999px;margin-bottom:6px;display:inline-block;border:1px solid rgba(26,163,154,.45)}

        /* Typing */
        .typing{display:none;max-width:840px;margin:0 auto;padding:0 16px}
        .typing.active{display:flex;gap:10px;animation:fadeIn .2s ease}
        .typing .avatar{width:30px;height:30px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:.75rem;flex-shrink:0;background:#3a2407;color:#ffd39a;font-weight:700;border:1px solid rgba(245,158,11,.5)}
        .typing-dots{display:flex;gap:4px;padding:10px 14px;background:rgba(23,36,55,.92);border-radius:12px;border:1px solid var(--line)}
        .typing-dots span{width:6px;height:6px;background:#8aa4c0;border-radius:50%;animation:bounce 1.25s infinite ease-in-out}
        .typing-dots span:nth-child(2){animation-delay:.2s}
        .typing-dots span:nth-child(3){animation-delay:.4s}
        @keyframes bounce{0%,80%,100%{transform:scale(.8);opacity:.45}40%{transform:scale(1);opacity:1}}

        /* Input */
        .input-area{padding:11px 16px 14px;border-top:1px solid var(--line);background:linear-gradient(180deg, rgba(13,21,33,.95), rgba(10,17,27,.92));flex-shrink:0}
        .input-container{max-width:840px;margin:0 auto}
        .attached-file{display:none;align-items:center;gap:6px;padding:7px 10px;background:rgba(26,163,154,.13);border:1px solid rgba(26,163,154,.42);border-radius:10px;margin-bottom:7px;font-size:.78rem;color:#b1fff6}
        .attached-file.visible{display:flex}
        .attached-file .af-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
        .attached-file .af-size{color:#8ee9de;font-size:.68rem}
        .attached-file .af-remove{background:none;border:none;color:#fdb4b4;cursor:pointer;font-size:1rem;padding:0 2px}
        .upload-progress{display:none;margin-bottom:7px;padding:8px 10px;border-radius:10px;border:1px solid rgba(26,163,154,.35);background:rgba(26,163,154,.10)}
        .upload-progress.visible{display:block}
        .upload-progress .up-meta{display:flex;justify-content:space-between;gap:8px;font-size:.74rem;color:#b9fff7;margin-bottom:5px}
        .upload-progress .up-ratio{font-size:.68rem;color:#90e8dd}
        .upload-progress .up-track{height:7px;border-radius:999px;background:rgba(14,24,38,.85);border:1px solid var(--line);overflow:hidden}
        .upload-progress .up-bar{height:100%;width:0%;background:linear-gradient(90deg,var(--brand),#1fbe9f);transition:width .16s ease}
        .active-scope{display:none;align-items:center;gap:8px;padding:7px 10px;background:rgba(245,158,11,.11);border:1px solid rgba(245,158,11,.35);border-radius:10px;margin-bottom:7px;font-size:.78rem;color:#ffe1af}
        .active-scope.visible{display:flex}
        .active-scope .scope-label{font-size:.68rem;color:#ffd089;text-transform:uppercase;letter-spacing:.04em}
        .active-scope .scope-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
        .active-scope .scope-clear{background:none;border:none;color:#fdb4b4;cursor:pointer;font-size:1rem;padding:0 2px}
        .input-row{display:flex;gap:7px;align-items:flex-end}
        .attach-btn{width:42px;height:42px;border-radius:11px;border:1px solid var(--line-strong);background:rgba(20,33,50,.84);color:#8fa7c5;font-size:1.1rem;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .15s}
        .attach-btn:hover{background:rgba(26,163,154,.15);color:#a3fff3;border-color:var(--brand)}
        .attach-btn.has-file{background:rgba(26,163,154,.18);color:#a3fff3;border-color:var(--brand)}
        .input-row textarea{flex:1;padding:10px 13px;background:rgba(20,32,48,.95);border:1px solid var(--line-strong);border-radius:12px;color:var(--text);font-size:.87rem;font-family:'IBM Plex Sans KR','Noto Sans KR',sans-serif;resize:none;min-height:42px;max-height:140px;outline:none;line-height:1.45}
        .input-row textarea:focus{border-color:var(--brand);box-shadow:0 0 0 3px rgba(26,163,154,.18)}
        .input-row textarea::placeholder{color:#748ba6}
        .send-btn{width:42px;height:42px;border-radius:11px;border:none;background:linear-gradient(120deg, var(--brand), #1fbe9f);color:#06251f;font-weight:700;font-size:1rem;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:.16s ease}
        .send-btn:hover{transform:translateY(-1px);box-shadow:0 8px 20px rgba(19,151,140,.28)}
        .send-btn:disabled{background:#405368;color:#a4b5c7;cursor:not-allowed;box-shadow:none}
        input[type="file"]{display:none}

        /* Docs drawer */
        .drawer-overlay{display:none;position:fixed;inset:0;background:rgba(6,10,17,.62);z-index:100}
        .drawer-overlay.open{display:block}
        .drawer{position:fixed;right:-360px;top:0;bottom:0;width:min(92vw,328px);background:linear-gradient(180deg, rgba(16,26,39,.98), rgba(10,17,27,.97));border-left:1px solid var(--line);z-index:101;transition:right .25s;display:flex;flex-direction:column;box-shadow:-20px 0 40px rgba(0,0,0,.35)}
        .drawer.open{right:0}
        .drawer-header{padding:15px 18px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}
        .drawer-header h3{font-family:'Space Grotesk','IBM Plex Sans KR',sans-serif;font-size:.92rem;color:#f6fcff;letter-spacing:.03em}
        .drawer-close{background:none;border:none;color:#8ea4bf;font-size:1.1rem;cursor:pointer}
        .drawer-body{flex:1;overflow-y:auto;padding:12px}
        .drawer-doc{padding:8px 10px;background:rgba(20,33,50,.85);border:1px solid var(--line);border-radius:8px;margin-bottom:4px;font-size:.76rem;display:flex;justify-content:space-between}
        .drawer-doc .name{color:#d7e8fb;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
        .drawer-doc .chunks{color:#92efe3;font-weight:600;margin-left:8px}
        .drawer-empty{color:#6f88a5;font-size:.78rem;text-align:center;padding:24px}
        .drawer-actions{padding:10px 14px;border-top:1px solid var(--line);display:flex;gap:6px}
        .drawer-actions button{flex:1;padding:7px;border-radius:8px;border:none;cursor:pointer;font-size:.72rem;font-weight:600}
        .btn-ref{background:#22354c;color:#d4e8ff}.btn-ref:hover{background:#2e4868}
        .btn-rst{background:#5b1a2a;color:#ffd1db}.btn-rst:hover{background:#7f243b}

        @keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
        @keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

        @media(max-width:980px){
            .sidebar{display:none}
            .topbar{padding:10px 12px}
            .topbar-center{order:3;flex:1 1 100%;justify-content:stretch}
            .model-select{width:100%}
            .status-badge#chunksBadge{display:none}
        }

        @media(max-width:768px){
            .messages{padding:14px 10px}
            .msg-wrap,.typing,.input-container{max-width:100%}
            .topbar{gap:8px}
            .topbar h1{font-size:.9rem}
            .topbar-right{gap:6px;max-width:100%;overflow-x:auto;padding-bottom:2px}
            .topbar-btn{padding:4px 7px}
            .input-area{padding:9px 10px 12px}
            .input-row textarea{font-size:.85rem}
        }
    </style>
</head>
<body class="theme-ocean">
    <!-- Sidebar: Chat History -->
    <aside class="sidebar">
        <div class="sidebar-top">
            <button class="new-chat-btn" onclick="newChat()">+ New Chat</button>
        </div>

        <div class="shelf-wrap">
            <div class="history-label">My Uploads</div>
            <div class="shelf-head">
                <button class="shelf-upload-btn" onclick="openShelfUpload()">+ Upload Files</button>
            </div>
            <div class="shelf-list" id="shelfList">
                <div class="shelf-empty">No uploaded files yet</div>
            </div>
            <input type="file" id="shelfFileInput" accept=".pdf,.png,.jpg,.jpeg,.webp" multiple>
        </div>

        <div class="history-label">Chats</div>
        <div class="history-list" id="historyList"></div>
        <div class="sidebar-footer">Tilon AI Chatbot v7.4</div>
    </aside>

    <main class="main">
        <!-- Topbar -->
        <div class="topbar">
            <h1>Tilon AI</h1>
            <div class="topbar-center">
                <select class="model-select" id="modelSelect"></select>
            </div>
            <div class="topbar-right">
                <span class="status-badge"><span class="dot" id="ollamaDot"></span><span id="ollamaStatus">...</span></span>
                <span class="status-badge" id="chunksBadge">0</span>
                <div class="control-stack">
                    <select class="theme-select" id="themeSelect" title="Theme">
                        <option value="ocean">Ocean</option>
                        <option value="rose">Rose</option>
                    </select>
                    <label class="web-toggle" for="webSearchToggle">
                        <input type="checkbox" id="webSearchToggle" checked> Web Search ON/OFF
                    </label>
                </div>
                <button class="topbar-btn" onclick="toggleDrawer()">Docs</button>
            </div>
        </div>

        <!-- Messages -->
        <div class="messages" id="messages"></div>

        <!-- Typing -->
        <div class="typing" id="typingIndicator">
            <div class="avatar">AI</div>
            <div class="typing-dots"><span></span><span></span><span></span></div>
        </div>

        <!-- Input -->
        <div class="input-area">
            <div class="input-container">
                <div class="attached-file" id="attachedFile">
                    <span id="afName"></span>
                    <span class="af-size" id="afSize"></span>
                    <button class="af-remove" onclick="removeFile()">&times;</button>
                </div>
                <div class="active-scope" id="activeScope">
                    <span class="scope-label">Scoped to</span>
                    <span class="scope-name" id="activeScopeName"></span>
                    <button class="scope-clear" onclick="clearActiveSource()">&times;</button>
                </div>
                <div class="upload-progress" id="uploadProgress">
                    <div class="up-meta">
                        <span id="upLabel">Uploading files: 0/0</span>
                        <span class="up-ratio" id="upRatio">0%</span>
                    </div>
                    <div class="up-track"><div class="up-bar" id="upBar"></div></div>
                </div>
                <div class="input-row">
                    <button class="attach-btn" id="attachBtn" onclick="document.getElementById('fileInput').click()">&#128206;</button>
                    <textarea id="chatInput" placeholder="Type a message or attach a file..." rows="1"
                        onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMessage();}"></textarea>
                    <button class="send-btn" id="sendBtn" onclick="sendMessage()">&#10148;</button>
                </div>
            </div>
            <input type="file" id="fileInput" accept=".pdf,.png,.jpg,.jpeg,.webp" multiple>
        </div>
    </main>

    <!-- Docs Drawer -->
    <div class="drawer-overlay" id="drawerOverlay" onclick="toggleDrawer()"></div>
    <div class="drawer" id="drawer">
        <div class="drawer-header"><h3>Stored Documents</h3><button class="drawer-close" onclick="toggleDrawer()">&times;</button></div>
        <div class="drawer-body" id="docList"><div class="drawer-empty">No documents</div></div>
        <div class="drawer-actions"><button class="btn-ref" onclick="loadDocs()">Refresh</button><button class="btn-rst" onclick="resetDB()">Reset DB</button></div>
    </div>

<script>
const messagesEl=document.getElementById('messages');
const chatInput=document.getElementById('chatInput');
const sendBtn=document.getElementById('sendBtn');
const attachBtn=document.getElementById('attachBtn');
const typingEl=document.getElementById('typingIndicator');
const fileInput=document.getElementById('fileInput');
const attachedFileEl=document.getElementById('attachedFile');
const afName=document.getElementById('afName');
const afSize=document.getElementById('afSize');
const uploadProgressEl=document.getElementById('uploadProgress');
const upLabel=document.getElementById('upLabel');
const upRatio=document.getElementById('upRatio');
const upBar=document.getElementById('upBar');
const activeScopeEl=document.getElementById('activeScope');
const activeScopeName=document.getElementById('activeScopeName');
const modelSelect=document.getElementById('modelSelect');
const themeSelect=document.getElementById('themeSelect');
const webSearchToggle=document.getElementById('webSearchToggle');
const historyList=document.getElementById('historyList');
const shelfList=document.getElementById('shelfList');
const shelfFileInput=document.getElementById('shelfFileInput');

let pendingFiles=[];
let currentChatId=null;
let chats={};  // {id: {title, messages: [{role,content,sources,mode,fileName}]}}
let chatsSyncTimer=null;
let applyingServerChats=false;

const CHAT_STORAGE_KEY='tilon_chats';
const CHAT_SYNC_DELAY_MS=350;

// ═══════════════════════════════════════════════════════════
// Chat History (local + server sync)
// ═══════════════════════════════════════════════════════════

function normalizeChatsShape(){
    if(!chats||typeof chats!=='object'||Array.isArray(chats)){
        chats={};
        return;
    }
    for(const id of Object.keys(chats)){
        if(!Array.isArray(chats[id].messages))chats[id].messages=[];
        if(typeof chats[id].activeSource!=='string')chats[id].activeSource='';
        if(typeof chats[id].activeDocId!=='string')chats[id].activeDocId='';
        if(typeof chats[id].title!=='string')chats[id].title='New Chat';
    }
}

function loadChats(){
    try{chats=JSON.parse(localStorage.getItem(CHAT_STORAGE_KEY)||'{}');}catch{chats={};}
    normalizeChatsShape();
}

function persistChatsLocalOnly(){
    try{localStorage.setItem(CHAT_STORAGE_KEY,JSON.stringify(chats));}catch{}
}

function queueChatsSync(){
    if(applyingServerChats)return;
    if(chatsSyncTimer)clearTimeout(chatsSyncTimer);
    chatsSyncTimer=setTimeout(async()=>{
        try{
            await fetch('/ui-state/chats',{
                method:'PUT',
                headers:{'Content-Type':'application/json'},
                body:JSON.stringify({chats}),
            });
        }catch{}
    },CHAT_SYNC_DELAY_MS);
}

function saveChats(){
    persistChatsLocalOnly();
    queueChatsSync();
}

async function loadChatsFromServer(){
    try{
        const resp=await fetch('/ui-state/chats');
        const data=await readApiJson(resp);
        if(!resp.ok)return;

        const serverChats=(data&&typeof data==='object'&&!Array.isArray(data)&&typeof data.chats==='object'&&!Array.isArray(data.chats))
            ? data.chats
            : {};

        if(!Object.keys(serverChats).length)return;

        applyingServerChats=true;
        chats=serverChats;
        normalizeChatsShape();
        persistChatsLocalOnly();

        const ids=Object.keys(chats).sort((a,b)=>parseInt(b.split('_')[1])-parseInt(a.split('_')[1]));
        if(ids.length>0)loadChat(ids[0]);
    }catch{}
    finally{applyingServerChats=false;}
}

function newChat(){
    currentChatId='chat_'+Date.now();
    chats[currentChatId]={title:'New Chat',messages:[],activeSource:'',activeDocId:''};
    saveChats();
    messagesEl.innerHTML='';
    renderActiveSource();
    renderHistory();
    renderShelfDocs(window.__shelfDocs||[]);
    chatInput.focus();
}

function loadChat(id){
    currentChatId=id;
    const chat=chats[id];
    if(!chat)return;
    messagesEl.innerHTML='';
    for(const m of chat.messages){
        appendMessageDOM(m.role,m.content,m.sources,m.mode,m.fileName);
    }
    renderActiveSource();
    renderHistory();
    renderShelfDocs(window.__shelfDocs||[]);
    scrollBottom();
}

function deleteChat(id,e){
    e.stopPropagation();
    delete chats[id];
    saveChats();
    if(currentChatId===id)newChat();
    else renderHistory();
}

function renderHistory(){
    const sorted=Object.entries(chats).sort((a,b)=>parseInt(b[0].split('_')[1])-parseInt(a[0].split('_')[1]));
    historyList.innerHTML='';
    for(const[id,chat]of sorted){
        const div=document.createElement('div');
        div.className='history-item'+(id===currentChatId?' active':'');
        div.onclick=()=>loadChat(id);
        div.innerHTML=`<span style="flex:1;overflow:hidden;text-overflow:ellipsis">${esc(chat.title)}</span><button class="del" onclick="deleteChat('${id}',event)">&times;</button>`;
        historyList.appendChild(div);
    }
}

function updateChatTitle(text){
    if(!currentChatId||!chats[currentChatId])return;
    if(chats[currentChatId].title==='New Chat'){
        chats[currentChatId].title=text.slice(0,40)+(text.length>40?'...':'');
        saveChats();renderHistory();
    }
}

function renderActiveSource(){
    const activeSource=(currentChatId&&chats[currentChatId])?chats[currentChatId].activeSource:'';
    if(activeSource){
        activeScopeName.textContent=activeSource;
        activeScopeEl.classList.add('visible');
    }else{
        activeScopeName.textContent='';
        activeScopeEl.classList.remove('visible');
    }
}

function setActiveSource(source){
    if(!currentChatId||!chats[currentChatId])return;
    chats[currentChatId].activeSource=source||'';
    if(!source)chats[currentChatId].activeDocId='';
    saveChats();
    renderActiveSource();
    renderShelfDocs(window.__shelfDocs||[]);
}

function setActiveDocument(source,docId){
    if(!currentChatId||!chats[currentChatId])return;
    chats[currentChatId].activeSource=source||'';
    chats[currentChatId].activeDocId=docId||'';
    saveChats();
    renderActiveSource();
    renderShelfDocs(window.__shelfDocs||[]);
}

function clearActiveSource(){
    if(!currentChatId||!chats[currentChatId]||!chats[currentChatId].activeSource)return;
    const cleared=chats[currentChatId].activeSource;
    setActiveSource('');
    appendMessageDOM('system','Document scope cleared: '+cleared);
    pushMessage('system','Document scope cleared: '+cleared);
}

function pushMessage(role,content,sources,mode,fileName){
    if(!currentChatId)newChat();
    chats[currentChatId].messages.push({role,content,sources:sources||[],mode:mode||'',fileName:fileName||''});
    saveChats();
}

// ═══════════════════════════════════════════════════════════
// Persistent Upload Shelf
// ═══════════════════════════════════════════════════════════

function openShelfUpload(){
    if(shelfFileInput)shelfFileInput.click();
}

function setScopeFromShelf(source,docId){
    if(!source)return;
    if(!currentChatId)newChat();
    setActiveDocument(source,docId||'');
    appendMessageDOM('system',`Scoped to uploaded file: ${source}`);
    pushMessage('system',`Scoped to uploaded file: ${source}`);
    renderShelfDocs(window.__shelfDocs||[]);
}

function renderShelfDocs(items){
    if(!shelfList)return;
    const docs=Array.isArray(items)?items:[];
    const activeSource=(currentChatId&&chats[currentChatId])?chats[currentChatId].activeSource:'';

    if(!docs.length){
        shelfList.innerHTML='<div class="shelf-empty">No uploaded files yet</div>';
        return;
    }

    shelfList.innerHTML='';
    for(const item of docs){
        const row=document.createElement('div');
        row.className='shelf-item';

        const nameBtn=document.createElement('button');
        nameBtn.type='button';
        nameBtn.className='shelf-item-name'+((activeSource&&activeSource===item.source)?' active':'');
        nameBtn.textContent=item.source||'unknown';
        nameBtn.title='Click to scope this chat to this file';
        nameBtn.onclick=()=>setScopeFromShelf(item.source||'',item.doc_id||'');

        const meta=document.createElement('span');
        meta.className='shelf-item-meta';
        const pagePart=item.pageTotal?` • ${item.pageTotal}p`:'';
        meta.textContent=`${item.chunks||0} chunks${pagePart}`;

        row.appendChild(nameBtn);
        row.appendChild(meta);
        shelfList.appendChild(row);
    }
}

async function loadShelfDocs(){
    if(!shelfList)return;
    try{
        const resp=await fetch('/docs-list');
        const data=await readApiJson(resp);
        if(!resp.ok)throw new Error(data.detail||'Failed to load docs');

        const grouped={};
        for(const d of (data.documents||[])){
            if((d.source_type||'')!=='upload')continue;
            const key=d.doc_id||`${d.source||'?'}::upload`;
            if(!grouped[key]){
                grouped[key]={
                    doc_id:d.doc_id||'',
                    source:d.source||'unknown',
                    pageTotal:d.page_total||'',
                    chunks:0,
                };
            }
            grouped[key].chunks+=1;
        }

        const docs=Object.values(grouped).sort((a,b)=>String(a.source).localeCompare(String(b.source),'ko'));
        window.__shelfDocs=docs;
        renderShelfDocs(docs);
    }catch{
        shelfList.innerHTML='<div class="shelf-empty">Failed to load uploaded files</div>';
    }
}

async function uploadToShelf(files){
    if(!files||!files.length)return;
    if(!currentChatId)newChat();

    sendBtn.disabled=true;
    typingEl.classList.add('active');
    setUploadProgress(0,files.length);

    let ok=0;
    let totalChunks=0;
    const failed=[];

    try{
        for(let i=0;i<files.length;i++){
            const file=files[i];
            setUploadProgress(i,files.length,file.name);
            try{
                const uploaded=await uploadFileForBatch(file);
                ok+=1;
                totalChunks+=(Number(uploaded.chunks_stored)||0);
            }catch(err){
                failed.push(file.name||`file-${i+1}`);
            }
            setUploadProgress(i+1,files.length,file.name);
        }

        const summary=`Shelf upload: ${ok}/${files.length} files ingested (${totalChunks} chunks).`;
        appendMessageDOM('system',summary);
        pushMessage('system',summary);

        if(failed.length){
            const preview=failed.slice(0,3).join(', ');
            const suffix=failed.length>3?' ...':'';
            const failMsg=`Failed files: ${preview}${suffix}`;
            appendMessageDOM('system',failMsg);
            pushMessage('system',failMsg);
        }

        await loadShelfDocs();
        loadHealth();
    }finally{
        typingEl.classList.remove('active');
        sendBtn.disabled=false;
        clearUploadProgress();
    }
}

if(shelfFileInput){
    shelfFileInput.addEventListener('change',async()=>{
        const files=Array.from(shelfFileInput.files||[]);
        shelfFileInput.value='';
        await uploadToShelf(files);
    });
}

// ═══════════════════════════════════════════════════════════
// File Attachment
// ═══════════════════════════════════════════════════════════
// File Attachment
// ═══════════════════════════════════════════════════════════

function fileKey(file){
    return [file.name,file.size,file.lastModified].join('::');
}

function renderPendingFiles(){
    if(pendingFiles.length===0){
        attachedFileEl.classList.remove('visible');
        attachBtn.classList.remove('has-file');
        afName.textContent='';
        afSize.textContent='';
        return;
    }

    if(pendingFiles.length===1){
        afName.textContent=pendingFiles[0].name;
        afSize.textContent=fmtBytes(pendingFiles[0].size);
    }else{
        const totalSize=pendingFiles.reduce((sum,f)=>sum+(f.size||0),0);
        afName.textContent=`${pendingFiles.length} files selected`;
        afSize.textContent=fmtBytes(totalSize);
    }

    attachedFileEl.classList.add('visible');
    attachBtn.classList.add('has-file');
}

fileInput.addEventListener('change',()=>{
    if(fileInput.files.length>0){
        const map=new Map(pendingFiles.map((f)=>[fileKey(f),f]));
        for(const f of Array.from(fileInput.files)){
            map.set(fileKey(f),f);
        }
        pendingFiles=Array.from(map.values());
        renderPendingFiles();
        chatInput.focus();
    }
    fileInput.value='';
});

function removeFile(){
    pendingFiles=[];
    renderPendingFiles();
}
function fmtBytes(b){
    if(b<1024)return b+' B';
    if(b<1048576)return(b/1024).toFixed(1)+' KB';
    return(b/1048576).toFixed(1)+' MB';
}
// ═══════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════
// Send Message
// ═══════════════════════════════════════════════════════════

async function readApiJson(resp){
    const raw=await resp.text();
    if(!raw)return {};
    try{
        return JSON.parse(raw);
    }catch{
        return {detail:raw};
    }
}

function setUploadProgress(done,total,currentFile=''){
    if(!uploadProgressEl||!upLabel||!upRatio||!upBar)return;
    const totalCount=Math.max(Number(total)||0,1);
    const doneCount=Math.max(0,Math.min(Number(done)||0,Number(total)||0));
    const percent=Math.round((doneCount/totalCount)*100);
    const filePart=currentFile?` · ${currentFile}`:'';

    upLabel.textContent=`Uploading files: ${doneCount}/${total}${filePart}`;
    upRatio.textContent=`${percent}%`;
    upBar.style.width=`${percent}%`;
    uploadProgressEl.classList.add('visible');
}

function clearUploadProgress(){
    if(!uploadProgressEl||!upLabel||!upRatio||!upBar)return;
    uploadProgressEl.classList.remove('visible');
    upLabel.textContent='Uploading files: 0/0';
    upRatio.textContent='0%';
    upBar.style.width='0%';
}

async function uploadFileForBatch(file){
    const fd=new FormData();
    fd.append('file',file);

    const resp=await fetch('/upload',{method:'POST',body:fd});
    const data=await readApiJson(resp);

    if(!resp.ok){
        throw new Error(data.detail||`${file.name} upload failed`);
    }

    return data;
}

async function sendMessage(){
    const text=chatInput.value.trim();
    const files=pendingFiles.slice();
    if(!text&&files.length===0)return;
    if(!currentChatId)newChat();

    const displayText=text||(files.length===1?'Analyze this document':`Analyze these ${files.length} documents`);
    const selectedModel=modelSelect.value;
    const webSearchEnabled=!!(webSearchToggle&&webSearchToggle.checked);
    const activeSource=chats[currentChatId].activeSource||'';
    const activeDocId=chats[currentChatId].activeDocId||'';

    const fileBadge=files.length===1?files[0].name:(files.length>1?`${files.length} files`:null);
    appendMessageDOM('user',displayText,null,null,fileBadge);
    pushMessage('user',displayText,null,null,fileBadge);
    updateChatTitle(displayText);

    chatInput.value='';chatInput.style.height='auto';
    removeFile();sendBtn.disabled=true;typingEl.classList.add('active');scrollBottom();

    try{
        let data=null;

        if(files.length===1){
            const file=files[0];
            const fd=new FormData();
            fd.append('file',file);
            fd.append('message',displayText);
            fd.append('model',selectedModel);
            fd.append('web_search_enabled',webSearchEnabled?'true':'false');

            const resp=await fetch('/chat-with-file',{method:'POST',body:fd});
            data=await readApiJson(resp);
            if(!resp.ok){showError(data.detail||'Upload failed');return;}

            setActiveDocument(data.active_source||file.name,data.active_doc_id||'');
            if(data.ingest&&data.ingest.count>0){
                const msg=file.name+' — '+data.ingest.count+' chunks ingested';
                appendMessageDOM('system',msg);
                pushMessage('system',msg);
            }
            await loadShelfDocs();
        }else if(files.length>1){
            setUploadProgress(0,files.length);

            const success=[];
            const failed=[];
            let totalChunks=0;

            for(let i=0;i<files.length;i++){
                const file=files[i];
                setUploadProgress(i,files.length,file.name);

                try{
                    const uploaded=await uploadFileForBatch(file);
                    const chunks=Number(uploaded.chunks_stored)||0;
                    totalChunks+=chunks;
                    success.push({
                        file:file.name,
                        chunks,
                        doc_id:uploaded.doc_id||'',
                        status:'success',
                    });
                }catch(err){
                    failed.push({
                        file:file.name,
                        reason:err?.message||'Upload failed',
                        status:'failed',
                    });
                }

                setUploadProgress(i+1,files.length,file.name);
            }

            const summary=`Multi upload: ${success.length}/${files.length} files ingested (${totalChunks} chunks).`;
            appendMessageDOM('system',summary);
            pushMessage('system',summary);

            if(failed.length){
                const failedNames=failed.slice(0,3).map(r=>r.file||'unknown').join(', ');
                const suffix=failed.length>3?' ...':'';
                const failMsg=`Failed files: ${failedNames}${suffix}`;
                appendMessageDOM('system',failMsg);
                pushMessage('system',failMsg);
            }

            setActiveSource('');
            await loadShelfDocs();

            if(text&&success.length>0){
                const history=chats[currentChatId].messages.filter(m=>m.role==='user'||m.role==='assistant').slice(-8);
                const resp=await fetch('/chat',{
                    method:'POST',headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({
                        message:text,
                        history:history,
                        model:selectedModel,
                        active_source:null,
                        active_doc_id:null,
                        web_search_enabled:webSearchEnabled
                    })
                });
                data=await readApiJson(resp);
                if(!resp.ok){showError(data.detail||'Error');return;}
                if(
                    Object.prototype.hasOwnProperty.call(data,'active_source')
                    || Object.prototype.hasOwnProperty.call(data,'active_doc_id')
                ){
                    setActiveDocument(data.active_source||'',data.active_doc_id||'');
                }
            }else if(text&&success.length===0){
                data={
                    answer:'No files were ingested successfully. Please check the file types and try again.',
                    sources:[],
                    mode:'document_qa'
                };
            }else{
                data={
                    answer:`${success.length} files are ready. Ask a question to test retrieval across documents.`,
                    sources:[],
                    mode:'document_qa'
                };
            }
        }else{
            const history=chats[currentChatId].messages.filter(m=>m.role==='user'||m.role==='assistant').slice(-8);
            const resp=await fetch('/chat',{
                method:'POST',headers:{'Content-Type':'application/json'},
                body:JSON.stringify({
                    message:text,
                    history:history,
                    model:selectedModel,
                    active_source:activeSource||null,
                    active_doc_id:activeDocId||null,
                    web_search_enabled:webSearchEnabled
                })
            });
            data=await readApiJson(resp);
            if(!resp.ok){showError(data.detail||'Error');return;}
            if(
                Object.prototype.hasOwnProperty.call(data,'active_source')
                || Object.prototype.hasOwnProperty.call(data,'active_doc_id')
            ){
                setActiveDocument(data.active_source||'',data.active_doc_id||'');
            }
        }

        if(data){
            appendMessageDOM('assistant',data.answer,data.sources,data.mode);
            pushMessage('assistant',data.answer,data.sources,data.mode);
        }
        loadHealth();
    }catch(err){showError('Connection error: '+err.message);}
    finally{typingEl.classList.remove('active');sendBtn.disabled=false;clearUploadProgress();chatInput.focus();scrollBottom();}
}
function showError(msg){
    appendMessageDOM('assistant','Error: '+msg);
    pushMessage('assistant','Error: '+msg);
    typingEl.classList.remove('active');sendBtn.disabled=false;
}

// ═══════════════════════════════════════════════════════════
// Render
// ═══════════════════════════════════════════════════════════

function appendMessageDOM(role,content,sources,mode,fileName){
    const av={user:'U',assistant:'AI',system:'i'};
    const nm={user:'You',assistant:'Tilon AI',system:'System'};
    let fileHtml=fileName?`<div class="file-badge">&#128206; ${esc(fileName)}</div>`:'';
    let modeHtml=mode?`<span class="mode-tag">${esc(mode)}</span>`:'';
    let srcHtml='';
    if(sources&&sources.length){
        srcHtml='<div class="sources">'+sources.map(s=>{
            const label=`${s.source||'?'} p.${s.page||'?'}`;
            const title=[s.doc_id||'',s.source_type||''].filter(Boolean).join(' | ');
            return `<span class="source-tag" title="${esc(title)}">${esc(label)}</span>`;
        }).join('')+'</div>';
    }
    const w=document.createElement('div');w.className='msg-wrap';
    w.innerHTML=`<div class="message ${role}"><div class="avatar">${av[role]||'?'}</div><div class="body"><div class="sender">${nm[role]||role} ${modeHtml}</div>${fileHtml}<div class="content">${esc(content)}</div>${srcHtml}</div></div>`;
    messagesEl.appendChild(w);scrollBottom();
}

function esc(t){const d=document.createElement('div');d.textContent=t;return d.innerHTML;}
function scrollBottom(){setTimeout(()=>{messagesEl.scrollTop=messagesEl.scrollHeight;},50);}
chatInput.addEventListener('input',()=>{chatInput.style.height='auto';chatInput.style.height=Math.min(chatInput.scrollHeight,140)+'px';});

// ═══════════════════════════════════════════════════════════
// Models
// ═══════════════════════════════════════════════════════════

async function loadModels(){
    try{
        const resp=await fetch('/models');
        const data=await readApiJson(resp);
        modelSelect.innerHTML='';
        const saved=localStorage.getItem('tilon_model');
        for(const m of(data.available||[])){
            const opt=document.createElement('option');
            opt.value=m.trim();opt.textContent=m.trim();
            if(saved&&m.trim()===saved)opt.selected=true;
            else if(!saved&&m.trim()===data.default)opt.selected=true;
            modelSelect.appendChild(opt);
        }
    }catch{modelSelect.innerHTML='<option>llama3.1:latest</option>';}
}

modelSelect.addEventListener('change',()=>{localStorage.setItem('tilon_model',modelSelect.value);});

const THEME_CLASS={ocean:'theme-ocean',rose:'theme-rose'};

function applyTheme(theme){
    const selected=Object.prototype.hasOwnProperty.call(THEME_CLASS,theme)?theme:'ocean';
    document.body.classList.remove(...Object.values(THEME_CLASS));
    document.body.classList.add(THEME_CLASS[selected]);
    if(themeSelect)themeSelect.value=selected;
    localStorage.setItem('tilon_theme',selected);
}

function initTheme(){
    const saved=localStorage.getItem('tilon_theme')||'ocean';
    applyTheme(saved);
}

if(themeSelect){
    themeSelect.addEventListener('change',()=>applyTheme(themeSelect.value));
}

function initWebSearchToggle(){
    if(!webSearchToggle)return;
    const saved=localStorage.getItem('tilon_web_search_enabled');
    const enabled=saved===null?true:saved==='true';
    webSearchToggle.checked=enabled;
}

if(webSearchToggle){
    webSearchToggle.addEventListener('change',()=>{
        localStorage.setItem('tilon_web_search_enabled',webSearchToggle.checked?'true':'false');
    });
}

// ═══════════════════════════════════════════════════════════
// Docs Drawer
// ═══════════════════════════════════════════════════════════

function toggleDrawer(){document.getElementById('drawer').classList.toggle('open');document.getElementById('drawerOverlay').classList.toggle('open');loadDocs();}

async function loadDocs(){
    const dl=document.getElementById('docList');
    try{
        const resp=await fetch('/docs-list');const data=await readApiJson(resp);
        if(!data.documents||!data.documents.length){dl.innerHTML='<div class="drawer-empty">No documents stored</div>';return;}
        const g={};
        for(const d of data.documents){
            const key=d.doc_id||`${d.source||'?'}::${d.source_type||''}`;
            if(!g[key])g[key]={source:d.source||'?',sourceType:d.source_type||'',chunks:0,pageTotal:d.page_total||''};
            g[key].chunks++;
        }
        dl.innerHTML='';
        for(const item of Object.values(g)){
            const subtitle=[item.sourceType,item.pageTotal?`${item.pageTotal}p`:'' ].filter(Boolean).join(' • ');
            const d=document.createElement('div');
            d.className='drawer-doc';
            d.innerHTML=`<div style="display:flex;flex-direction:column;min-width:0;flex:1"><span class="name">${esc(item.source)}</span>${subtitle?`<span style="font-size:.68rem;color:#94a3b8">${esc(subtitle)}</span>`:''}</div><span class="chunks">${item.chunks}</span>`;
            dl.appendChild(d);
        }
    }catch{dl.innerHTML='<div class="drawer-empty">Error</div>';}
}

async function resetDB(){
    if(!confirm('Reset vector DB? All documents deleted.'))return;
    try{
        await fetch('/reset-db',{method:'DELETE'});
        for(const id of Object.keys(chats)){
            chats[id].activeSource='';
            chats[id].activeDocId='';
        }
        saveChats();
        renderActiveSource();
        appendMessageDOM('system','Vector DB reset.');
        pushMessage('system','Vector DB reset.');
        loadDocs();loadShelfDocs();loadHealth();
    }
    catch(err){appendMessageDOM('system','Reset failed');}
}

// ═══════════════════════════════════════════════════════════
// Health
// ═══════════════════════════════════════════════════════════

async function loadHealth(){
    try{
        const resp=await fetch('/health');const d=await readApiJson(resp);
        document.getElementById('ollamaDot').className='dot '+(d.ollama==='connected'?'green':'red');
        document.getElementById('ollamaStatus').textContent=d.ollama==='connected'?'Online':'Offline';
        document.getElementById('chunksBadge').textContent=(d.documents_in_vectorstore||0)+' chunks';
    }catch{document.getElementById('ollamaDot').className='dot red';document.getElementById('ollamaStatus').textContent='Err';}
}

// ═══════════════════════════════════════════════════════════
// Init
// ═══════════════════════════════════════════════════════════

loadChats();
initTheme();
initWebSearchToggle();
loadModels();
loadHealth();
loadShelfDocs();
setInterval(loadHealth,30000);

// Load most recent chat or start new
const chatIds=Object.keys(chats).sort((a,b)=>parseInt(b.split('_')[1])-parseInt(a.split('_')[1]));
if(chatIds.length>0){loadChat(chatIds[0]);}else{newChat();}

loadChatsFromServer();
chatInput.focus();
</script>
</body>
</html>
"""

@router.get("/ui", response_class=HTMLResponse)
def chat_ui():
    if UI_INDEX_PATH.exists():
        return FileResponse(UI_INDEX_PATH)
    logger.warning("static UI not found at %s, serving embedded fallback UI", UI_INDEX_PATH)
    return HTMLResponse(CHAT_UI_HTML)
