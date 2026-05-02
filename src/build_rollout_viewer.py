"""
Build an interactive GRPO rollout explorer.

Visualizes multiple rollouts per maze with:
  - All G rollouts side by side with paths on the maze grid
  - Step-by-step path animation
  - Reward decomposition bars
  - Advantage visualization (which rollouts get reinforced)
  - Comparison across training stages

Usage:
    uv run python -m src.build_rollout_viewer \
        --rollouts results/rollouts.json \
        --output results/rollout_viewer.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# noqa: E501 — HTML template contains long lines

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GRPO Rollout Explorer</title>
<style>
:root {
  --bg: #0f1117; --fg: #e4e4e7; --accent: #6366f1;
  --green: #22c55e; --red: #ef4444; --amber: #f59e0b; --blue: #3b82f6;
  --surface: #1c1e26; --border: #2e3039;
  --cell: 24px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--fg); font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace; font-size: 13px; }

.header { padding: 16px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: baseline; gap: 20px; }
.header h1 { font-size: 16px; font-weight: 600; color: var(--accent); }
.header .stats { font-size: 12px; color: #888; }

.controls { padding: 10px 24px; border-bottom: 1px solid var(--border); display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.controls label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
.controls select, .controls input { background: var(--surface); border: 1px solid var(--border); color: var(--fg); padding: 5px 8px; border-radius: 4px; font-family: inherit; font-size: 12px; }

.main { display: flex; height: calc(100vh - 100px); }

.list { width: 280px; border-right: 1px solid var(--border); overflow-y: auto; flex-shrink: 0; }
.list-item { padding: 8px 12px; border-bottom: 1px solid var(--border); cursor: pointer; font-size: 12px; }
.list-item:hover { background: var(--surface); }
.list-item.active { background: #1e1b4b; border-left: 3px solid var(--accent); }
.list-item .top { display: flex; justify-content: space-between; align-items: center; }
.list-item .id { font-weight: 600; }
.badge { padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: 600; }
.badge.solved { background: #052e16; color: var(--green); }
.badge.failed { background: #2a0a0a; color: var(--red); }
.badge.partial { background: #1a1500; color: var(--amber); }
.list-item .sub { color: #666; margin-top: 2px; }
.reward-bar-mini { height: 3px; border-radius: 2px; margin-top: 4px; background: var(--border); }
.reward-bar-mini .fill { height: 100%; border-radius: 2px; }

.detail { flex: 1; overflow-y: auto; padding: 20px 24px; }

.section { margin-bottom: 24px; }
.section h3 { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; }

.maze-and-correct { display: flex; gap: 24px; align-items: flex-start; margin-bottom: 20px; }
.maze-info h2 { font-size: 18px; font-weight: 600; margin-bottom: 4px; }
.maze-info .meta { color: #888; font-size: 12px; margin-bottom: 12px; }

.maze-grid { display: inline-grid; gap: 0; border: 2px solid var(--border); border-radius: 4px; }
.maze-cell { width: var(--cell); height: var(--cell); display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 700; transition: background 0.15s; }
.maze-cell.wall { background: #374151; }
.maze-cell.open { background: #1f2937; }
.maze-cell.entry { background: #1e3a5f; color: #60a5fa; }
.maze-cell.exit { background: #1e3a5f; color: #60a5fa; }
.maze-cell.correct-path { background: #052e16; color: var(--green); }
.maze-cell.model-path { background: #1e1b4b; color: var(--accent); }
.maze-cell.both-path { background: #1a2e05; color: var(--green); outline: 2px solid var(--accent); outline-offset: -2px; }
.maze-cell.current-step { background: var(--amber) !important; color: #000 !important; }

.legend { display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 4px; font-size: 11px; color: #888; }
.legend-swatch { width: 12px; height: 12px; border-radius: 2px; }

/* Rollout cards */
.rollouts-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
.rollout-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px; cursor: pointer; transition: border-color 0.15s; }
.rollout-card:hover { border-color: var(--accent); }
.rollout-card.selected { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.rollout-card.best { border-color: var(--green); }
.rollout-card.worst { border-color: var(--red); }
.rollout-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.rollout-label { font-weight: 600; font-size: 12px; }
.rollout-reward { font-size: 14px; font-weight: 700; }

/* Advantage bar */
.advantage-bar { height: 20px; display: flex; align-items: center; margin: 6px 0; border-radius: 3px; background: var(--border); position: relative; }
.advantage-fill { height: 100%; border-radius: 3px; position: absolute; }
.advantage-fill.positive { background: var(--green); left: 50%; }
.advantage-fill.negative { background: var(--red); right: 50%; }
.advantage-label { position: relative; z-index: 1; width: 100%; text-align: center; font-size: 10px; font-weight: 600; }

/* Reward breakdown */
.reward-breakdown { display: flex; gap: 8px; margin-top: 6px; }
.reward-component { flex: 1; }
.reward-component .label { font-size: 10px; color: #888; }
.reward-component .bar { height: 4px; border-radius: 2px; background: var(--border); margin-top: 2px; }
.reward-component .bar .fill { height: 100%; border-radius: 2px; }

.moves-display { font-size: 12px; letter-spacing: 1px; margin-top: 6px; }
.move-char { padding: 1px 3px; border-radius: 2px; margin: 0 1px; font-size: 11px; }
.move-valid { background: #052e16; color: var(--green); }
.move-invalid { background: #2a0a0a; color: var(--red); }

/* Animation controls */
.anim-controls { display: flex; gap: 8px; align-items: center; margin-top: 8px; }
.anim-controls button { background: var(--surface); border: 1px solid var(--border); color: var(--fg); padding: 4px 12px; border-radius: 4px; cursor: pointer; font-family: inherit; font-size: 12px; }
.anim-controls button:hover { border-color: var(--accent); }
.anim-controls .step-label { font-size: 12px; color: #888; min-width: 80px; }
</style>
</head>
<body>
<div class="header">
  <h1>GRPO Rollout Explorer</h1>
  <div class="stats" id="stats"></div>
</div>
<div class="controls">
  <div><label>Size</label> <select id="filter-size"><option value="all">All</option></select></div>
  <div><label>Sort</label> <select id="sort-by"><option value="id">ID</option><option value="reward-desc">Reward ↓</option><option value="reward-asc">Reward ↑</option><option value="variance">Variance ↓</option></select></div>
</div>
<div class="main">
  <div class="list" id="list"></div>
  <div class="detail" id="detail"><p style="color:#666;padding:40px;">Select a maze from the list to explore its rollouts.</p></div>
</div>

<script>
const DATA = __DATA_PLACEHOLDER__;
let filtered = [...DATA];
let selected = null;
let animStep = -1;
let animTimer = null;
let selectedRollout = 0;

function init() {
  const sizes = [...new Set(DATA.map(d => d.width+'x'+d.height))].sort();
  const sel = document.getElementById('filter-size');
  sizes.forEach(s => { const o = document.createElement('option'); o.value=s; o.textContent=s; sel.appendChild(o); });

  const totalMazes = DATA.length;
  const totalRollouts = DATA.reduce((s,d) => s + d.rollouts.length, 0);
  const anySolved = DATA.filter(d => d.rollouts.some(r => r.solved)).length;
  document.getElementById('stats').textContent =
    totalMazes+' mazes | '+totalRollouts+' rollouts | '+anySolved+' with ≥1 solve';

  document.getElementById('filter-size').addEventListener('change', applyFilters);
  document.getElementById('sort-by').addEventListener('change', applyFilters);
  applyFilters();
}

function applyFilters() {
  const size = document.getElementById('filter-size').value;
  const sort = document.getElementById('sort-by').value;
  filtered = DATA.filter(d => size==='all' || (d.width+'x'+d.height)===size);
  if (sort==='reward-desc') filtered.sort((a,b)=>b.reward_mean-a.reward_mean);
  else if (sort==='reward-asc') filtered.sort((a,b)=>a.reward_mean-b.reward_mean);
  else if (sort==='variance') filtered.sort((a,b)=>b.reward_std-a.reward_std);
  else filtered.sort((a,b)=>a.maze_id.localeCompare(b.maze_id));
  renderList();
}

function renderList() {
  const list = document.getElementById('list');
  list.innerHTML = '';
  filtered.forEach(d => {
    const best = Math.max(...d.rollouts.map(r=>r.reward));
    const anySolved = d.rollouts.some(r=>r.solved);
    const el = document.createElement('div');
    el.className = 'list-item' + (selected===d.maze_id?' active':'');
    const badgeCls = anySolved ? 'solved' : best > 0 ? 'partial' : 'failed';
    const badgeText = anySolved ? 'SOLVED' : best > 0 ? 'PARTIAL' : 'FAILED';
    const pct = Math.max(0, Math.min(100, (d.reward_mean+1)/2*100));
    const barColor = anySolved ? 'var(--green)' : best>0 ? 'var(--amber)' : 'var(--red)';
    el.innerHTML =
      '<div class="top"><span class="id">'+d.maze_id+'</span><span class="badge '+badgeCls+'">'+badgeText+'</span></div>'+
      '<div class="sub">μ='+d.reward_mean.toFixed(3)+' σ='+d.reward_std.toFixed(3)+' | '+d.solution_length+' moves</div>'+
      '<div class="reward-bar-mini"><div class="fill" style="width:'+pct+'%;background:'+barColor+';"></div></div>';
    el.onclick = () => { selected=d.maze_id; selectedRollout=0; animStep=-1; clearInterval(animTimer); renderList(); renderDetail(d); };
    list.appendChild(el);
  });
}

function renderMazeGrid(d, rolloutIdx, step) {
  const gridRows = 2*d.height+1, gridCols = 2*d.width+1;
  const mazeLines = d.maze_str.split('\\n');
  const correctCells = new Set(d.correct_path.map(p=>p[0]+','+p[1]));
  const ro = d.rollouts[rolloutIdx];
  const showPath = step < 0 ? ro.path : ro.path.slice(0, step+1);
  const modelCells = new Set(showPath.map(p=>p[0]+','+p[1]));
  const currentCell = step >= 0 && step < ro.path.length ? ro.path[step][0]+','+ro.path[step][1] : null;

  let html = '<div class="maze-grid" style="grid-template-columns:repeat('+gridCols+',var(--cell));">';
  for (let r=0; r<gridRows; r++) {
    const chars = mazeLines[r] ? mazeLines[r].split(' ') : [];
    for (let c=0; c<gridCols; c++) {
      const ch = chars[c]||'#';
      const isOddR=r%2===1, isOddC=c%2===1;
      const cellR=Math.floor(r/2), cellC=Math.floor(c/2);
      const cellKey=cellR+','+cellC;
      const isCell=isOddR&&isOddC;
      const inCorrect=isCell&&correctCells.has(cellKey);
      const inModel=isCell&&modelCells.has(cellKey);

      let cls='maze-cell', display=ch==='.'?'':ch;
      if (ch==='#') cls+=' wall';
      else if (ch==='>') { cls+=' entry'; display='>'; }
      else cls+=' open';

      if (isCell && cellKey===currentCell) { cls='maze-cell current-step'; display='▶'; }
      else if (isCell && inCorrect && inModel) { cls='maze-cell both-path'; display='●'; }
      else if (isCell && inCorrect) { cls='maze-cell correct-path'; display='○'; }
      else if (isCell && inModel) { cls='maze-cell model-path'; display='●'; }

      if (isOddR && c===0 && ch==='>') { cls='maze-cell entry'; display='>'; }
      if (isOddR && c===gridCols-1 && ch==='>') { cls='maze-cell exit'; display='>'; }

      html += '<div class="'+cls+'">'+display+'</div>';
    }
  }
  html += '</div>';
  return html;
}

function renderDetail(d) {
  const detail = document.getElementById('detail');
  const ro = d.rollouts[selectedRollout];
  const maxAdv = Math.max(...d.advantages.map(Math.abs), 0.01);
  const bestIdx = d.rollouts.reduce((bi,r,i,a) => r.reward > a[bi].reward ? i : bi, 0);
  const worstIdx = d.rollouts.reduce((wi,r,i,a) => r.reward < a[wi].reward ? i : wi, 0);

  let html = '';

  // Maze info + correct solution grid
  html += '<div class="maze-and-correct">';
  html += '<div class="maze-info"><h2>'+d.maze_id+'</h2>';
  html += '<div class="meta">'+d.width+'×'+d.height+' | Solution: '+d.solution_length+' moves | '+
    d.rollouts.length+' rollouts | μ='+d.reward_mean.toFixed(3)+' σ='+d.reward_std.toFixed(3)+'</div>';
  html += '<div class="legend">';
  html += '<div class="legend-item"><div class="legend-swatch" style="background:#374151;"></div>Wall</div>';
  html += '<div class="legend-item"><div class="legend-swatch" style="background:#052e16;"></div>Correct</div>';
  html += '<div class="legend-item"><div class="legend-swatch" style="background:#1e1b4b;"></div>Model</div>';
  html += '<div class="legend-item"><div class="legend-swatch" style="background:var(--amber);"></div>Current step</div>';
  html += '</div></div>';

  // Selected rollout maze
  html += '<div><h3 style="margin-bottom:8px;">Rollout #'+(selectedRollout+1)+
    (selectedRollout===bestIdx?' (best)':'')+(selectedRollout===worstIdx?' (worst)':'')+'</h3>';
  html += renderMazeGrid(d, selectedRollout, animStep);
  html += '<div class="anim-controls">';
  html += '<button onclick="animReset()">⏮</button>';
  html += '<button onclick="animPrev()">◀</button>';
  html += '<button onclick="animToggle()">▶ Play</button>';
  html += '<button onclick="animNext()">▶▷</button>';
  html += '<span class="step-label">Step: '+(animStep<0?'all':animStep+'/'+ro.path.length)+'</span>';
  html += '</div></div>';
  html += '</div>';

  // Correct moves
  html += '<div class="section"><h3>Correct Solution</h3>';
  html += '<div class="moves-display">'+d.correct_moves.map(m=>'<span class="move-char move-valid">'+m+'</span>').join(' ')+'</div></div>';

  // All rollouts as cards
  html += '<div class="section"><h3>All Rollouts — click to select</h3>';
  html += '<div class="rollouts-grid">';
  d.rollouts.forEach((r, i) => {
    const adv = d.advantages[i];
    const advPct = Math.abs(adv)/maxAdv*50;
    const advDir = adv >= 0 ? 'positive' : 'negative';
    const advStyle = adv >= 0
      ? 'left:50%;width:'+advPct+'%'
      : 'right:50%;width:'+advPct+'%';

    let cardCls = 'rollout-card';
    if (i===selectedRollout) cardCls += ' selected';
    if (i===bestIdx) cardCls += ' best';
    if (i===worstIdx && bestIdx!==worstIdx) cardCls += ' worst';

    const validMoves = (r.moves_parsed||[]).map((m,mi) => {
      const cls = mi < r.valid_steps ? 'move-char move-valid' : 'move-char move-invalid';
      return '<span class="'+cls+'">'+m+'</span>';
    }).join(' ');

    html += '<div class="'+cardCls+'" onclick="selectRollout('+i+')">';
    html += '<div class="rollout-header">';
    html += '<span class="rollout-label">#'+(i+1)+(r.solved?' ✓':'')+' '+(i===bestIdx?'⭐':'')+'</span>';
    html += '<span class="rollout-reward" style="color:'+(r.solved?'var(--green)':r.reward>0?'var(--amber)':'var(--red)')+'">'+r.reward.toFixed(3)+'</span>';
    html += '</div>';

    // Advantage bar
    html += '<div class="advantage-bar">';
    html += '<div class="advantage-fill '+advDir+'" style="'+advStyle+'"></div>';
    html += '<div class="advantage-label">'+(adv>=0?'+':'')+adv.toFixed(2)+' advantage</div>';
    html += '</div>';

    // Reward breakdown
    const coverage = Math.min(r.valid_steps / Math.max(d.solution_length,1), 1);
    html += '<div class="reward-breakdown">';
    html += '<div class="reward-component"><div class="label">Coverage '+(coverage*100).toFixed(0)+'%</div><div class="bar"><div class="fill" style="width:'+(coverage*100)+'%;background:var(--blue);"></div></div></div>';
    html += '<div class="reward-component"><div class="label">Progress '+(r.progress*100).toFixed(0)+'%</div><div class="bar"><div class="fill" style="width:'+(r.progress*100)+'%;background:var(--amber);"></div></div></div>';
    html += '<div class="reward-component"><div class="label">Steps '+r.valid_steps+'/'+d.solution_length+'</div><div class="bar"><div class="fill" style="width:'+(Math.min(r.valid_steps/Math.max(d.solution_length,1),1)*100)+'%;background:var(--green);"></div></div></div>';
    html += '</div>';

    // Moves
    html += '<div class="moves-display">'+(validMoves||'<span style="color:#666;">no moves</span>')+'</div>';
    html += '</div>';
  });
  html += '</div></div>';

  // GRPO explanation
  html += '<div class="section" style="background:var(--surface);padding:16px;border-radius:6px;border:1px solid var(--border);">';
  html += '<h3 style="color:var(--accent);margin-bottom:8px;">How GRPO Uses These Rollouts</h3>';
  html += '<p style="color:#aaa;line-height:1.6;">';
  html += '1. <strong>Generate</strong> '+d.rollouts.length+' completions for this maze<br>';
  html += '2. <strong>Score</strong> each with the reward function (shown above)<br>';
  html += '3. <strong>Normalize</strong> rewards within the group: advantage = (reward - μ) / σ<br>';
  html += '&nbsp;&nbsp;&nbsp;μ = '+d.reward_mean.toFixed(3)+', σ = '+d.reward_std.toFixed(3)+'<br>';
  html += '4. <strong>Update</strong> policy: increase probability of high-advantage rollouts (⭐), decrease low-advantage ones<br>';
  if (d.reward_std < 0.005) {
    html += '<br><span style="color:var(--red);">⚠ All rollouts scored the same — advantages are ~0 and GRPO learns nothing from this maze!</span>';
  } else if (d.reward_std < 0.05) {
    html += '<br><span style="color:var(--amber);">△ Very low reward variance (σ='+d.reward_std.toFixed(3)+') — weak learning signal.</span>';
  }
  html += '</p></div>';

  detail.innerHTML = html;
}

function selectRollout(i) {
  selectedRollout = i;
  animStep = -1;
  clearInterval(animTimer);
  const d = DATA.find(d => d.maze_id === selected);
  if (d) renderDetail(d);
}

function animReset() { animStep = -1; clearInterval(animTimer); rerender(); }
function animPrev() { if (animStep > 0) animStep--; else animStep = -1; clearInterval(animTimer); rerender(); }
function animNext() {
  const d = DATA.find(d => d.maze_id === selected);
  if (!d) return;
  const maxStep = d.rollouts[selectedRollout].path.length - 1;
  if (animStep < maxStep) { if (animStep < 0) animStep = 0; else animStep++; }
  clearInterval(animTimer);
  rerender();
}
function animToggle() {
  const d = DATA.find(d => d.maze_id === selected);
  if (!d) return;
  if (animTimer) { clearInterval(animTimer); animTimer = null; return; }
  if (animStep < 0) animStep = 0;
  const maxStep = d.rollouts[selectedRollout].path.length - 1;
  animTimer = setInterval(() => {
    if (animStep >= maxStep) { clearInterval(animTimer); animTimer = null; return; }
    animStep++;
    rerender();
  }, 400);
}
function rerender() {
  const d = DATA.find(d => d.maze_id === selected);
  if (d) renderDetail(d);
}

init();
if (filtered.length > 0) { selected = filtered[0].maze_id; renderDetail(filtered[0]); renderList(); }
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Build rollout explorer")
    parser.add_argument("--rollouts", type=str, required=True,
                        help="Rollouts JSON from rollout_capture")
    parser.add_argument("--output", type=str, default="results/rollout_viewer.html")
    args = parser.parse_args()

    print(f"Loading rollouts from {args.rollouts}")
    with open(args.rollouts) as f:
        data = json.load(f)

    serialized = json.dumps(data).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", serialized)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    total_rollouts = sum(len(d["rollouts"]) for d in data)
    print(f"Viewer saved to {output_path} ({len(data)} mazes, {total_rollouts} rollouts)")


if __name__ == "__main__":
    main()
