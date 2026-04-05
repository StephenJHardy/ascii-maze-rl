"""
Build an interactive HTML maze explorer from eval results.

Combines eval results with maze data to produce a self-contained HTML file
that visualizes:
  - The maze grid with walls and passages
  - The correct solution path
  - The model's attempted path
  - Reward breakdown and move details

Usage:
    uv run python -m src.build_viewer \
        --results results/sft_large_eval_full.json \
        --dataset data/eval_full.jsonl \
        --output results/viewer.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.maze_dataset import MazeDataset
from src.maze_verify import simulate


def build_viewer_data(results_path: str, dataset_path: str) -> list[dict]:
    """Combine eval results with maze data for the viewer."""
    with open(results_path) as f:
        eval_data = json.load(f)

    dataset = MazeDataset.load(dataset_path)
    records_by_id = {r.id: r for r in dataset}

    viewer_entries = []
    for result in eval_data["results"]:
        record = records_by_id.get(result["maze_id"])
        if record is None:
            continue

        maze = record.to_maze()

        model_moves = result["moves_parsed"] or []
        model_path_tuples = simulate(model_moves, maze)
        model_path = [list(p) for p in model_path_tuples]

        correct_path = [list(p) for p in maze.solution]
        correct_moves = list(maze.solution_moves)

        viewer_entries.append({
            "maze_id": result["maze_id"],
            "width": record.width,
            "height": record.height,
            "maze_str": record.maze_str,
            "walls": record.walls,
            "entry": record.entry,
            "exit": record.exit,
            "correct_path": correct_path,
            "correct_moves": correct_moves,
            "solution_length": record.solution_length,
            "model_completion": result["completion"],
            "model_moves": model_moves,
            "model_path": model_path,
            "reward": result["reward"],
            "solved": result["solved"],
            "valid_steps": result["valid_steps"],
            "progress": result["progress"],
        })

    return viewer_entries


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ASCII Maze RL — Result Explorer</title>
<style>
  :root {
    --bg: #0f1117; --fg: #e4e4e7; --accent: #6366f1;
    --green: #22c55e; --red: #ef4444; --amber: #f59e0b;
    --surface: #1c1e26; --border: #2e3039;
    --cell: 28px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--fg); font-family: 'JetBrains Mono', 'Fira Code', monospace; }

  .header { padding: 20px 32px; border-bottom: 1px solid var(--border); display: flex; align-items: baseline; gap: 24px; }
  .header h1 { font-size: 18px; font-weight: 600; color: var(--accent); }
  .header .stats { font-size: 13px; color: #888; }

  .controls { padding: 12px 32px; border-bottom: 1px solid var(--border); display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  .controls label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
  .controls select, .controls input { background: var(--surface); border: 1px solid var(--border); color: var(--fg); padding: 6px 10px; border-radius: 4px; font-family: inherit; font-size: 13px; }

  .main { display: flex; height: calc(100vh - 110px); }

  .list { width: 340px; border-right: 1px solid var(--border); overflow-y: auto; }
  .list-item { padding: 10px 16px; border-bottom: 1px solid var(--border); cursor: pointer; font-size: 13px; display: flex; justify-content: space-between; align-items: center; }
  .list-item:hover { background: var(--surface); }
  .list-item.active { background: #1e1b4b; border-left: 3px solid var(--accent); }
  .list-item .id { font-weight: 500; }
  .list-item .badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
  .badge.solved { background: #052e16; color: var(--green); }
  .badge.failed { background: #2a0a0a; color: var(--red); }
  .list-item .reward { font-size: 12px; color: #888; }

  .detail { flex: 1; overflow-y: auto; padding: 24px 32px; }

  .detail-header { display: flex; gap: 24px; align-items: baseline; margin-bottom: 20px; }
  .detail-header h2 { font-size: 20px; font-weight: 600; }
  .detail-header .meta { font-size: 13px; color: #888; }

  .grid-container { display: flex; gap: 40px; margin-bottom: 24px; flex-wrap: wrap; }
  .grid-section { }
  .grid-section h3 { font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }

  .maze-grid { display: inline-grid; gap: 0; border: 2px solid var(--border); border-radius: 4px; }
  .maze-cell { width: var(--cell); height: var(--cell); display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; }
  .maze-cell.wall { background: #374151; }
  .maze-cell.open { background: #1f2937; }
  .maze-cell.entry { background: #1e3a5f; color: #60a5fa; }
  .maze-cell.exit { background: #1e3a5f; color: #60a5fa; }
  .maze-cell.correct-path { background: #052e16; color: var(--green); }
  .maze-cell.model-path { background: #1e1b4b; color: var(--accent); }
  .maze-cell.both-path { background: #1a2e05; color: var(--green); border: 2px solid var(--accent); }
  .maze-cell.model-collision { background: #2a0a0a; color: var(--red); }

  .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
  .info-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 14px; }
  .info-card h4 { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .info-card .value { font-size: 16px; font-weight: 600; }
  .info-card .sub { font-size: 12px; color: #888; margin-top: 4px; }

  .output-section { margin-bottom: 20px; }
  .output-section h3 { font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
  .output-box { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px; font-size: 13px; white-space: pre-wrap; word-break: break-all; max-height: 120px; overflow-y: auto; }
  .moves-display { font-size: 15px; letter-spacing: 2px; }
  .move-char { padding: 2px 4px; border-radius: 3px; margin: 0 1px; }
  .move-valid { background: #052e16; color: var(--green); }
  .move-invalid { background: #2a0a0a; color: var(--red); }
  .move-wasted { background: #1c1e26; color: #666; }

  .legend { display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #888; }
  .legend-swatch { width: 16px; height: 16px; border-radius: 3px; }
</style>
</head>
<body>
<div class="header">
  <h1>ASCII Maze RL — Result Explorer</h1>
  <div class="stats" id="stats"></div>
</div>
<div class="controls">
  <div><label>Size</label> <select id="filter-size"><option value="all">All</option></select></div>
  <div><label>Status</label> <select id="filter-status"><option value="all">All</option><option value="solved">Solved</option><option value="failed">Failed</option></select></div>
  <div><label>Sort</label> <select id="sort-by"><option value="id">ID</option><option value="reward-desc">Reward ↓</option><option value="reward-asc">Reward ↑</option><option value="length">Solution length</option></select></div>
  <div><label>Search</label> <input id="search" type="text" placeholder="maze id..."></div>
</div>
<div class="main">
  <div class="list" id="list"></div>
  <div class="detail" id="detail"><p style="color:#666;padding:40px;">Select a maze from the list.</p></div>
</div>

<script>
const DATA = __DATA_PLACEHOLDER__;

let filtered = [...DATA];
let selected = null;

function init() {
  const sizes = [...new Set(DATA.map(d => d.width + 'x' + d.height))].sort();
  const sizeSelect = document.getElementById('filter-size');
  sizes.forEach(s => { const o = document.createElement('option'); o.value = s; o.textContent = s; sizeSelect.appendChild(o); });

  const solved = DATA.filter(d => d.solved).length;
  document.getElementById('stats').textContent =
    DATA.length + ' mazes | ' + solved + ' solved (' + (100*solved/DATA.length).toFixed(1) + '%) | ' +
    sizes.join(', ');

  document.getElementById('filter-size').addEventListener('change', applyFilters);
  document.getElementById('filter-status').addEventListener('change', applyFilters);
  document.getElementById('sort-by').addEventListener('change', applyFilters);
  document.getElementById('search').addEventListener('input', applyFilters);

  applyFilters();
}

function applyFilters() {
  const size = document.getElementById('filter-size').value;
  const status = document.getElementById('filter-status').value;
  const sort = document.getElementById('sort-by').value;
  const search = document.getElementById('search').value.toLowerCase();

  filtered = DATA.filter(d => {
    if (size !== 'all' && (d.width + 'x' + d.height) !== size) return false;
    if (status === 'solved' && !d.solved) return false;
    if (status === 'failed' && d.solved) return false;
    if (search && !d.maze_id.toLowerCase().includes(search)) return false;
    return true;
  });

  if (sort === 'reward-desc') filtered.sort((a,b) => b.reward - a.reward);
  else if (sort === 'reward-asc') filtered.sort((a,b) => a.reward - b.reward);
  else if (sort === 'length') filtered.sort((a,b) => a.solution_length - b.solution_length);
  else filtered.sort((a,b) => a.maze_id.localeCompare(b.maze_id));

  renderList();
}

function renderList() {
  const list = document.getElementById('list');
  list.innerHTML = '';
  filtered.forEach((d, i) => {
    const el = document.createElement('div');
    el.className = 'list-item' + (selected === d.maze_id ? ' active' : '');
    el.innerHTML =
      '<div><span class="id">' + d.maze_id + '</span><br>' +
      '<span class="reward">reward: ' + d.reward.toFixed(3) + ' | ' + d.solution_length + ' moves</span></div>' +
      '<span class="badge ' + (d.solved ? 'solved' : 'failed') + '">' + (d.solved ? 'SOLVED' : 'FAILED') + '</span>';
    el.onclick = () => selectMaze(d);
    list.appendChild(el);
  });
}

function selectMaze(d) {
  selected = d.maze_id;
  renderList();
  renderDetail(d);
}

function renderDetail(d) {
  const detail = document.getElementById('detail');
  const gridRows = 2 * d.height + 1;
  const gridCols = 2 * d.width + 1;

  const correctCells = new Set(d.correct_path.map(p => p[0]+','+p[1]));
  const modelCells = new Set(d.model_path.map(p => p[0]+','+p[1]));

  function renderGrid(title, showCorrect, showModel) {
    const mazeLines = d.maze_str.split('\\n');
    let html = '<div class="grid-section"><h3>' + title + '</h3>';
    html += '<div class="maze-grid" style="grid-template-columns: repeat('+gridCols+', var(--cell));">';
    for (let r = 0; r < gridRows; r++) {
      const chars = mazeLines[r] ? mazeLines[r].split(' ') : [];
      for (let c = 0; c < gridCols; c++) {
        const ch = chars[c] || '#';
        const isOddR = r % 2 === 1, isOddC = c % 2 === 1;
        const cellR = Math.floor(r/2), cellC = Math.floor(c/2);
        const cellKey = cellR+','+cellC;
        const isCell = isOddR && isOddC;
        const inCorrect = isCell && showCorrect && correctCells.has(cellKey);
        const inModel = isCell && showModel && modelCells.has(cellKey);

        let cls = 'maze-cell';
        let display = ch === '.' ? '' : ch;
        if (ch === '#') cls += ' wall';
        else if (ch === '>') { cls += ' entry'; display = '>'; }
        else cls += ' open';

        if (isCell) {
          if (inCorrect && inModel) { cls = 'maze-cell both-path'; display = '●'; }
          else if (inCorrect) { cls = 'maze-cell correct-path'; display = '○'; }
          else if (inModel) { cls = 'maze-cell model-path'; display = '●'; }
        }

        // Check for entry/exit markers
        if (isOddR && c === 0 && ch === '>') { cls = 'maze-cell entry'; display = '>'; }
        if (isOddR && c === gridCols-1 && ch === '>') { cls = 'maze-cell exit'; display = '>'; }

        html += '<div class="'+cls+'">'+display+'</div>';
      }
    }
    html += '</div></div>';
    return html;
  }

  // Color the moves
  const validCount = d.valid_steps;
  const movesHtml = (d.model_moves || []).map((m, i) => {
    const cls = i < validCount ? 'move-char move-valid' : 'move-char move-invalid';
    return '<span class="'+cls+'">'+m+'</span>';
  }).join(' ');

  const correctMovesHtml = d.correct_moves.map(m =>
    '<span class="move-char move-valid">'+m+'</span>'
  ).join(' ');

  detail.innerHTML =
    '<div class="detail-header">' +
      '<h2>' + d.maze_id + '</h2>' +
      '<div class="meta">' + d.width + '×' + d.height + ' | ' +
        '<span style="color:' + (d.solved ? 'var(--green)' : 'var(--red)') + ';">' +
        (d.solved ? 'SOLVED' : 'FAILED') + '</span></div>' +
    '</div>' +

    '<div class="legend">' +
      '<div class="legend-item"><div class="legend-swatch" style="background:#374151;"></div> Wall</div>' +
      '<div class="legend-item"><div class="legend-swatch" style="background:#052e16;"></div> Correct path</div>' +
      '<div class="legend-item"><div class="legend-swatch" style="background:#1e1b4b;"></div> Model path</div>' +
      '<div class="legend-item"><div class="legend-swatch" style="background:#1a2e05;border:2px solid var(--accent);"></div> Both</div>' +
    '</div>' +

    '<div class="grid-container">' +
      renderGrid('Correct Solution', true, false) +
      renderGrid('Model Output', false, true) +
      renderGrid('Overlay', true, true) +
    '</div>' +

    '<div class="info-grid">' +
      '<div class="info-card"><h4>Reward</h4><div class="value">' + d.reward.toFixed(3) + '</div></div>' +
      '<div class="info-card"><h4>Progress</h4><div class="value">' + (d.progress*100).toFixed(1) + '%</div></div>' +
      '<div class="info-card"><h4>Valid Steps</h4><div class="value">' + d.valid_steps + ' / ' + (d.model_moves||[]).length + ' parsed</div>' +
        '<div class="sub">Solution needs ' + d.solution_length + ' moves</div></div>' +
      '<div class="info-card"><h4>Solution Length</h4><div class="value">' + d.solution_length + ' moves</div>' +
        '<div class="sub">Model output ' + (d.model_moves||[]).length + ' moves</div></div>' +
    '</div>' +

    '<div class="output-section"><h3>Correct Moves</h3>' +
      '<div class="output-box moves-display">' + correctMovesHtml + '</div></div>' +

    '<div class="output-section"><h3>Model Moves</h3>' +
      '<div class="output-box moves-display">' + (movesHtml || '<span style="color:#666;">no moves parsed</span>') + '</div></div>' +

    '<div class="output-section"><h3>Raw Model Output</h3>' +
      '<div class="output-box">' + escapeHtml(d.model_completion) + '</div></div>';
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

init();
if (filtered.length > 0) selectMaze(filtered[0]);
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Build maze result viewer")
    parser.add_argument("--results", type=str, required=True, help="Eval results JSON")
    parser.add_argument("--dataset", type=str, required=True, help="Eval dataset JSONL")
    parser.add_argument("--output", type=str, default="results/viewer.html")
    args = parser.parse_args()

    print(f"Loading results from {args.results}")
    print(f"Loading dataset from {args.dataset}")

    entries = build_viewer_data(args.results, args.dataset)
    print(f"Built {len(entries)} viewer entries")

    serialized = json.dumps(entries).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", serialized)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Viewer saved to {output_path} ({output_path.stat().st_size / 1000:.0f} KB)")


if __name__ == "__main__":
    main()
