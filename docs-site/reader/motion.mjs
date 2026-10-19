export function spring(update, initial=0) {
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  let x=initial, velocity=0, target=initial, frame=0, last=0, settle=()=>{};
  function step(now) {
    const dt=Math.min((now-last)/1000,1/30);last=now;
    const acceleration=180*(target-x)-20*velocity;
    velocity+=acceleration*dt;x+=velocity*dt;update(x);
    if(Math.abs(target-x)<.2&&Math.abs(velocity)<.2){x=target;velocity=0;frame=0;update(x);settle();}
    else frame=requestAnimationFrame(step);
  }
  return {to(value,done=()=>{}) {target=value;settle=done;
    if(reduced.matches){if(frame)cancelAnimationFrame(frame);frame=0;x=target;velocity=0;update(x);done();return;}
    if(!frame){last=performance.now();frame=requestAnimationFrame(step);}
  },finish(){if(frame)cancelAnimationFrame(frame);frame=0;x=target;velocity=0;update(x);settle();}};
}
