// Pure functions copied from chess.jsx (without React dependencies)
const col = p => p ? p[0] : null;
const typ = p => p ? p[1] : null;
const inB = (r,c) => r>=0&&r<8&&c>=0&&c<8;

const pseudoMoves = (board, r, c, ep, cas) => {
  const p=board[r][c]; if(!p) return [];
  const pc=col(p), pt=typ(p), en=pc==='w'?'b':'w';
  const moves=[];
  const add=(nr,nc,flag)=>{ if(inB(nr,nc)&&col(board[nr][nc])!==pc) moves.push([nr,nc,flag||null]); };
  const slide=(dirs)=>{ for(const[dr,dc]of dirs){let nr=r+dr,nc=c+dc; while(inB(nr,nc)){if(col(board[nr][nc])===pc)break; moves.push([nr,nc,null]); if(col(board[nr][nc])===en)break; nr+=dr;nc+=dc;}} };

  if(pt==='P'){
    const d=pc==='w'?-1:1, sr=pc==='w'?6:1;
    if(inB(r+d,c)&&!board[r+d][c]){ moves.push([r+d,c,null]); if(r===sr&&!board[r+2*d][c]) moves.push([r+2*d,c,null]); }
    for(const dc of[-1,1]){
      if(inB(r+d,c+dc)){
        if(col(board[r+d][c+dc])===en) moves.push([r+d,c+dc,null]);
        if(ep&&ep[0]===r+d&&ep[1]===c+dc) moves.push([r+d,c+dc,'ep']);
      }
    }
  } else if(pt==='N'){
    for(const[dr,dc]of[[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]]) add(r+dr,c+dc);
  } else if(pt==='B') slide([[-1,-1],[-1,1],[1,-1],[1,1]]);
  else if(pt==='R') slide([[-1,0],[1,0],[0,-1],[0,1]]);
  else if(pt==='Q') slide([[-1,-1],[-1,1],[1,-1],[1,1],[-1,0],[1,0],[0,-1],[0,1]]);
  else if(pt==='K'){
    for(const[dr,dc]of[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]) add(r+dr,c+dc);
    if(cas){
      const row=pc==='w'?7:0;
      if(r===row&&c===4){
        if((pc==='w'?cas.wK:cas.bK)&&!board[row][5]&&!board[row][6]) moves.push([row,6,'ck']);
        if((pc==='w'?cas.wQ:cas.bQ)&&!board[row][3]&&!board[row][2]&&!board[row][1]) moves.push([row,2,'cq']);
      }
    }
  }
  return moves;
};

const isAttacked = (board, r, c, byCol) => {
  for(let rr=0;rr<8;rr++) for(let cc=0;cc<8;cc++)
    if(col(board[rr][cc])===byCol && pseudoMoves(board,rr,cc,null,null).some(([mr,mc])=>mr===r&&mc===c)) return true;
  return false;
};

const findKing = (board, pc) => {
  for(let r=0;r<8;r++) for(let c=0;c<8;c++) if(board[r][c]===pc+'K') return [r,c];
  return null;
};

const applyMove = (board, [fr,fc], [tr,tc,flag], promo='Q') => {
  const nb=board.map(row=>[...row]);
  const p=nb[fr][fc], pc=col(p);
  nb[tr][tc]=p; nb[fr][fc]=null;
  if(flag==='ep'){ const cr=pc==='w'?tr+1:tr-1; nb[cr][tc]=null; }
  if(flag==='ck'){ nb[tr][5]=nb[tr][7]; nb[tr][7]=null; }
  if(flag==='cq'){ nb[tr][3]=nb[tr][0]; nb[tr][0]=null; }
  if(typ(p)==='P'&&(tr===0||tr===7)) nb[tr][tc]=pc+promo;
  return nb;
};

const legalMoves = (board, r, c, ep, cas) => {
  const p=board[r][c]; if(!p) return [];
  const pc=col(p), en=pc==='w'?'b':'w';
  return pseudoMoves(board,r,c,ep,cas).filter(move=>{
    const[tr,tc,flag]=move;
    if(flag==='ck'){ const row=pc==='w'?7:0; if(isAttacked(board,row,4,en)||isAttacked(board,row,5,en)||isAttacked(board,row,6,en)) return false; }
    if(flag==='cq'){ const row=pc==='w'?7:0; if(isAttacked(board,row,4,en)||isAttacked(board,row,3,en)||isAttacked(board,row,2,en)) return false; }
    const nb=applyMove(board,[r,c],move);
    const [kr,kc]=findKing(nb,pc);
    return !isAttacked(nb,kr,kc,en);
  });
};

// Agent configuration
const agentColor = 'b'; // Agent plays black

// Piece values for evaluation (king value set to 0 as it doesn't change in material count)
const pieceValue = {P:1, N:3, B:3, R:5, Q:9, K:0};

// Evaluate board from agent's perspective (black): positive means black is ahead
const evaluateBoard = (board) => {
  let score = 0;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const piece = board[r][c];
      if (!piece) continue;
      const color = col(piece);
      const type = typ(piece);
      const value = pieceValue[type] || 0;
      if (color === agentColor) {
        score += value;
      } else {
        score -= value;
      }
    }
  }
  return score;
};

// Get all legal moves for the current state
// State: { board, turn, ep, cas }
// Returns: array of { from: [r, c], move: [tr, tc, flag] }
const getAllLegalMoves = (state) => {
  const moves = [];
  const { board, turn, ep, cas } = state;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      if (col(board[r][c]) === turn) {
        const pieceMoves = legalMoves(board, r, c, ep, cas);
        pieceMoves.forEach(move => {
          moves.push({ from: [r, c], move });
        });
      }
    }
  }
  return moves;
};

// Apply a move to the state and return a new state
// move: { from: [fr, fc], move: [tr, tc, flag] }
// Returns: { board, turn, ep, cas }
const makeMove = (state, move) => {
  const { board, turn, ep, cas } = state;
  const { from, move: [tr, tc, flag] } = move;
  const newBoard = applyMove(board, from, [tr, tc, flag]);

  // Update ep: en passant target
  const p = board[from[0]][from[1]];
  const newEp = (typ(p)==='P' && Math.abs(tr - from[0]) === 2) ? [(from[0] + tr) / 2, tc] : null;

  // Update cas: castling rights
  const newCas = {...cas};
  if (p === 'wK') {newCas.wK = false; newCas.wQ = false;}
  if (p === 'bK') {newCas.bK = false; newCas.bQ = false;}
  if (from[0] === 7 && from[1] === 7) newCas.wK = false;
  if (from[0] === 7 && from[1] === 0) newCas.wQ = false;
  if (from[0] === 0 && from[1] === 7) newCas.bK = false;
  if (from[0] === 0 && from[1] === 0) newCas.bQ = false;

  // Update turn
  const newTurn = turn === 'w' ? 'b' : 'w';

  return { board: newBoard, turn: newTurn, ep: newEp, cas: newCas };
};

// Check if the game is over (checkmate or stalemate)
const isGameOver = (state) => {
  const { board, turn, ep, cas } = state;
  return ['checkmate', 'stalemate'].includes(gameStatus(board, turn, ep, cas));
};

// Minimax with alpha-beta pruning
// Returns the evaluation of the state from the agent's perspective
const minimax = (state, depth, alpha, beta, maximizingPlayer) => {
  if (depth === 0 || isGameOver(state)) {
    return evaluateBoard(state.board);
  }

  const legalMoves = getAllLegalMoves(state);
  if (maximizingPlayer) {
    let maxEval = -Infinity;
    for (const move of legalMoves) {
      const newState = makeMove(state, move);
      // After making a move, the turn has switched in newState
      // So the next player is the opponent if we are maximizing, and vice versa
      const eval = minimax(newState, depth - 1, alpha, beta, false);
      maxEval = Math.max(maxEval, eval);
      alpha = Math.max(alpha, eval);
      if (beta <= alpha) break;
    }
    return maxEval;
  } else {
    let minEval = Infinity;
    for (const move of legalMoves) {
      const newState = makeMove(state, move);
      const eval = minimax(newState, depth - 1, alpha, beta, true);
      minEval = Math.min(minEval, eval);
      beta = Math.min(beta, eval);
      if (beta <= alpha) break;
    }
    return minEval;
  }
};

// Get the best move for the agent (black) given the current state
// Returns: { from: [fr, fc], to: [tr, tc] } (without flag) or null if no moves
const getAgentMove = (state) => {
  // If it's not the agent's turn, return null (should not be called in that case)
  if (state.turn !== agentColor) {
    return null;
  }

  const legalMoves = getAllLegalMoves(state);
  if (legalMoves.length === 0) {
    return null; // No legal moves (game over)
  }

  let bestMove = null;
  let bestValue = -Infinity;
  let alpha = -Infinity;
  let beta = Infinity;

  // We are the maximizing player (black)
  for (const move of legalMoves) {
    const newState = makeMove(state, move);
    // After our move, it's the opponent's turn (white) -> minimizing player
    const moveValue = minimax(newState, 2, alpha, beta, false); // depth 2
    if (moveValue > bestValue) {
      bestValue = moveValue;
      bestMove = move;
      alpha = Math.max(alpha, bestValue);
    }
  }

  if (bestMove) {
    // Return in the format expected by the bridge: { from, to: [tr, tc] } (without flag)
    return {
      from: bestMove.from,
      to: bestMove.move.slice(0, 2) // [tr, tc] without flag
    };
  }
  return null;
};

// Export for use in the component
export { getAgentMove };
