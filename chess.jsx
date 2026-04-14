import { useState, useCallback, useEffect } from "react";

const PIECES = {
  wK:"♔",wQ:"♕",wR:"♖",wB:"♗",wN:"♘",wP:"♙",
  bK:"♚",bQ:"♛",bR:"♜",bB:"♝",bN:"♞",bP:"♟",
};

const col = p => p ? p[0] : null;
const typ = p => p ? p[1] : null;
const inB = (r,c) => r>=0&&r<8&&c>=0&&c<8;

const initBoard = () => {
  const b = Array(8).fill(null).map(()=>Array(8).fill(null));
  b[0]=['bR','bN','bB','bQ','bK','bB','bN','bR'];
  b[1]=Array(8).fill('bP');
  b[6]=Array(8).fill('wP');
  b[7]=['wR','wN','wB','wQ','wK','wB','wN','wR'];
  return b;
};

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

const gameStatus = (board, turn, ep, cas) => {
  const en=turn==='w'?'b':'w';
  const [kr,kc]=findKing(board,turn);
  const inChk=isAttacked(board,kr,kc,en);
  for(let r=0;r<8;r++) for(let c=0;c<8;c++)
    if(col(board[r][c])===turn && legalMoves(board,r,c,ep,cas).length>0) return inChk?'check':'playing';
  return inChk?'checkmate':'stalemate';
};

const SQUARE = 72;

export default function Chess() {
  const [board,setBoard]=useState(initBoard());
  const [sel,setSel]=useState(null);
  const [valid,setValid]=useState([]);
  const [turn,setTurn]=useState('w');
  const [ep,setEp]=useState(null);
  const [cas,setCas]=useState({wK:true,wQ:true,bK:true,bQ:true});
  const [promo,setPromo]=useState(null);
  const [capW,setCapW]=useState([]);
  const [capB,setCapB]=useState([]);
  const [status,setStatus]=useState('playing');
  const [lastMove,setLastMove]=useState(null);
  const [flipped,setFlipped]=useState(false);

  const reset = () => {
    setBoard(initBoard()); setSel(null); setValid([]); setTurn('w');
    setEp(null); setCas({wK:true,wQ:true,bK:true,bQ:true});
    setPromo(null); setCapW([]); setCapB([]); setStatus('playing'); setLastMove(null);
  };

  const execMove = useCallback((from, move, promoPiece=null) => {
    const[fr,fc]=from, [tr,tc,flag]=move;
    const p=board[fr][fc], pc=col(p);
    if(typ(p)==='P'&&(tr===0||tr===7)&&!promoPiece){ setPromo({from,move}); return; }
    const nb=applyMove(board,from,move,promoPiece||'Q');
    const captured=board[tr][tc];
    if(captured){ col(captured)==='w'?setCapW(v=>[...v,captured]):setCapB(v=>[...v,captured]); }
    if(flag==='ep'){ const cr=pc==='w'?tr+1:tr-1; const epc=board[cr][tc]; col(epc)==='w'?setCapW(v=>[...v,epc]):setCapB(v=>[...v,epc]); }
    const newEp=(typ(p)==='P'&&Math.abs(tr-fr)===2)?[(fr+tr)/2,tc]:null;
    const newCas={...cas};
    if(p==='wK'){newCas.wK=false;newCas.wQ=false;} if(p==='bK'){newCas.bK=false;newCas.bQ=false;}
    if(fr===7&&fc===7)newCas.wK=false; if(fr===7&&fc===0)newCas.wQ=false;
    if(fr===0&&fc===7)newCas.bK=false; if(fr===0&&fc===0)newCas.bQ=false;
    const nt=pc==='w'?'b':'w';
    setBoard(nb); setTurn(nt); setEp(newEp); setCas(newCas);
    setSel(null); setValid([]); setPromo(null); setLastMove([from,move]);
    setStatus(gameStatus(nb,nt,newEp,newCas));
  }, [board,cas]);

  // --- RL INTERACTION BRIDGE ---
  useEffect(() => {
    window.chess = {
      state: {
        board,
        turn,
        status,
        captured: { white: capW, black: capB },
        lastMove
      },
      getLegalMoves: () => {
        const moves = [];
        for (let r = 0; r < 8; r++) {
          for (let c = 0; c < 8; c++) {
            if (col(board[r][c]) === turn) {
              const mvs = legalMoves(board, r, c, ep, cas);
              mvs.forEach(m => moves.push({ from: [r, c], to: m }));
            }
          }
        }
        return moves;
      },
      move: (fr, fc, tr, tc) => {
        const mvs = legalMoves(board, fr, fc, ep, cas);
        const target = mvs.find(([mr, mc]) => mr === tr && mc === tc);
        if (target) {
          execMove([fr, fc], target);
          return true;
        }
        return false;
      },
      reset: () => reset()
    };
  }, [board, turn, status, capW, capB, ep, cas, execMove]);
  // -----------------------------

  const handleClick = (r, c) => {
    if(status==='checkmate'||status==='stalemate') return;
    if(promo) return;
    if(sel){
      const mv=valid.find(([mr,mc])=>mr===r&&mc===c);
      if(mv){ execMove(sel,mv); return; }
    }
    const p=board[r][c];
    if(p&&col(p)===turn){ setSel([r,c]); setValid(legalMoves(board,r,c,ep,cas)); }
    else { setSel(null); setValid([]); }
  };

  const [ekr,ekc]=findKing(board,turn)||[-1,-1];
  const inCheck=(status==='check'||status==='checkmate')&&isAttacked(board,ekr,ekc,turn==='w'?'b':'w');

  const ranks=flipped?[0,1,2,3,4,5,6,7]:[7,6,5,4,3,2,1,0];
  const files=flipped?[7,6,5,4,3,2,1,0]:[0,1,2,3,4,5,6,7];

  const pieceValue={P:1,N:3,B:3,R:5,Q:9};
  const score=(side)=>{
    const opp=side==='w'?'b':'w';
    const cap=side==='w'?capB:capW;
    const oppCap=side==='w'?capW:capB;
    const adv=cap.reduce((s,p)=>s+(pieceValue[typ(p)]||0),0)-oppCap.reduce((s,p)=>s+(pieceValue[typ(p)]||0),0);
    return adv>0?`+${adv}`:'';
  };

  return (
    <div style={{display:'flex',flexDirection:'column',alignItems:'center',padding:'20px 16px',minHeight:'100vh',background:'#1c1510',fontFamily:"'Georgia',serif",userSelect:'none'}}>
      <div style={{color:'#c9a84c',fontSize:'1.6rem',letterSpacing:'0.25em',marginBottom:'4px',fontWeight:'normal',textTransform:'uppercase'}}>Chess</div>

      {/* Status bar */}
      <div style={{
        fontSize:'0.82rem',letterSpacing:'0.12em',textTransform:'uppercase',marginBottom:'16px',
        color:status==='checkmate'?'#e05555':status==='stalemate'?'#999':status==='check'?'#e09055':'#9a8060',
        minHeight:'18px'
      }}>
        {status==='checkmate'&&`${turn==='w'?'Black':'White'} wins by checkmate`}
        {status==='stalemate'&&'Stalemate — Draw'}
        {status==='check'&&`${turn==='w'?'White':'Black'} is in check!`}
        {status==='playing'&&`${turn==='w'?'White':'Black'} to move`}
      </div>

      {/* Black's captured pieces + score */}
      <div style={{width:SQUARE*8,display:'flex',alignItems:'center',marginBottom:'6px',minHeight:'22px',gap:'2px'}}>
        {capW.map((p,i)=><span key={i} style={{fontSize:'15px',color:'#ddd',lineHeight:1}}>{PIECES[p]}</span>)}
        <span style={{marginLeft:'6px',fontSize:'13px',color:'#9a8060',fontFamily:'monospace'}}>{score('b')}</span>
      </div>

      {/* Board */}
      <div style={{position:'relative',boxShadow:'0 12px 40px rgba(0,0,0,0.8)',border:'3px solid #6b4e1a',lineHeight:0}}>
        {/* Rank labels */}
        <div style={{position:'absolute',right:'100%',top:0,bottom:0,paddingRight:'6px',display:'flex',flexDirection:'column',justifyContent:'space-around',pointerEvents:'none'}}>
          {ranks.map(r=><span key={r} style={{fontSize:'11px',color:'#6b4e1a',lineHeight:1,textAlign:'right'}}>{r+1}</span>)}
        </div>
        {/* File labels */}
        <div style={{position:'absolute',top:'100%',left:0,right:0,paddingTop:'5px',display:'flex',justifyContent:'space-around',pointerEvents:'none'}}>
          {files.map(c=><span key={c} style={{fontSize:'11px',color:'#6b4e1a',lineHeight:1,textAlign:'center',width:SQUARE+'px'}}>{String.fromCharCode(97+c)}</span>)}
        </div>

        <div>
          {ranks.map((r)=>(
            <div key={r} style={{display:'flex'}}>
              {files.map((c)=>{
                const isLight=(r+c)%2===0;
                const isSel=sel&&sel[0]===r&&sel[1]===c;
                const isVM=valid.some(([mr,mc])=>mr===r&&mc===c);
                const isCapVM=isVM&&!!board[r][c]&&col(board[r][c])!==turn;
                const isEpVM=isVM&&!board[r][c]&&ep&&ep[0]===r&&ep[1]===c;
                const isLast=lastMove&&((lastMove[0][0]===r&&lastMove[0][1]===c)||(lastMove[1][0]===r&&lastMove[1][1]===c));
                const isKingChk=inCheck&&r===ekr&&c===ekc;
                const piece=board[r][c];
                let bg=isLight?'#f0d9b5':'#b58863';
                if(isKingChk) bg='#c84040';
                else if(isSel) bg=isLight?'#f6f669':'#cdd16a';
                else if(isLast) bg=isLight?'#cdd26a':'#aaa23a';
                return (
                  <div key={c} onClick={()=>handleClick(r,c)} style={{
                    width:SQUARE,height:SQUARE,background:bg,
                    display:'flex',alignItems:'center',justifyContent:'center',
                    cursor:piece&&col(piece)===turn&&status!=='checkmate'&&status!=='stalemate'?'pointer':isVM?'pointer':'default',
                    position:'relative',boxSizing:'border-box',
                  }}>
                    {/* Move dot */}
                    {isVM&&!isCapVM&&!isEpVM&&<div style={{position:'absolute',width:24,height:24,borderRadius:'50%',background:'rgba(0,0,0,0.22)',pointerEvents:'none'}}/>}
                    {/* Capture ring */}
                    {(isCapVM||isEpVM)&&<div style={{position:'absolute',inset:3,border:'4px solid rgba(0,0,0,0.22)',borderRadius:'50%',pointerEvents:'none'}}/>}
                    {/* Piece */}
                    {piece&&<span style={{
                      fontSize:44,lineHeight:1,display:'block',
                      filter:col(piece)==='w'?'drop-shadow(0 1px 2px rgba(0,0,0,0.35))':'drop-shadow(0 1px 2px rgba(0,0,0,0.6))',
                      position:'relative',zIndex:1,
                      transform:isSel?'scale(1.12)':'scale(1)',
                      transition:'transform 0.1s ease',
                    }}>{PIECES[piece]}</span>}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* White's captured pieces + score */}
      <div style={{width:SQUARE*8,display:'flex',alignItems:'center',marginTop:'6px',minHeight:'22px',gap:'2px'}}>
        {capB.map((p,i)=><span key={i} style={{fontSize:'15px',color:'#fff',lineHeight:1}}>{PIECES[p]}</span>)}
        <span style={{marginLeft:'6px',fontSize:'13px',color:'#9a8060',fontFamily:'monospace'}}>{score('w')}</span>
      </div>

      {/* Promotion modal */}
      {promo&&(
        <div style={{position:'fixed',inset:0,background:'rgba(0,0,0,0.75)',display:'flex',alignItems:'center',justifyContent:'center',zIndex:999}}>
          <div style={{background:'#2a1f0a',border:'2px solid #c9a84c',borderRadius:'10px',padding:'28px 32px',textAlign:'center'}}>
            <div style={{color:'#c9a84c',marginBottom:'18px',fontSize:'0.9rem',letterSpacing:'0.15em',textTransform:'uppercase'}}>Promote pawn</div>
            <div style={{display:'flex',gap:'12px'}}>
              {['Q','R','B','N'].map(pp=>(
                <button key={pp} onClick={()=>execMove(promo.from,promo.move,pp)} style={{
                  background:'#f0d9b5',border:'2px solid #c9a84c',borderRadius:'6px',
                  padding:'10px 14px',cursor:'pointer',fontSize:'40px',lineHeight:1,
                  transition:'transform 0.1s',
                }} onMouseOver={e=>e.currentTarget.style.transform='scale(1.1)'}
                  onMouseOut={e=>e.currentTarget.style.transform='scale(1)'}>
                  {PIECES[turn+''+pp]}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Buttons */}
      <div style={{display:'flex',gap:'12px',marginTop:'20px'}}>
        <button onClick={reset} style={{
          background:'transparent',border:'1px solid #6b4e1a',color:'#c9a84c',
          padding:'8px 22px',cursor:'pointer',letterSpacing:'0.18em',fontSize:'0.78rem',
          textTransform:'uppercase',borderRadius:'3px',transition:'background 0.2s',fontFamily:"'Georgia',serif",
        }} onMouseOver={e=>e.currentTarget.style.background='rgba(201,168,76,0.1)'}
          onMouseOut={e=>e.currentTarget.style.background='transparent'}>New Game</button>
        <button onClick={()=>setFlipped(f=>!f)} style={{
          background:'transparent',border:'1px solid #6b4e1a',color:'#c9a84c',
          padding:'8px 22px',cursor:'pointer',letterSpacing:'0.18em',fontSize:'0.78rem',
          textTransform:'uppercase',borderRadius:'3px',transition:'background 0.2s',fontFamily:"'Georgia',serif",
        }} onMouseOver={e=>e.currentTarget.style.background='rgba(201,168,76,0.1)'}
          onMouseOut={e=>e.currentTarget.style.background='transparent'}>Flip Board</button>
      </div>

      {/* Legend */}
      <div style={{marginTop:'16px',fontSize:'0.72rem',color:'#5a4a30',letterSpacing:'0.08em',textAlign:'center',lineHeight:1.8}}>
        Click a piece to select · Click a dot to move · Highlighted squares show valid moves
      </div>
    </div>
  );
}