import {fork} from 'node:child_process';
export async function withServer(script,environment,run){
  const server=fork(script,[],{env:{...process.env,...environment},stdio:['inherit','inherit','inherit','ipc']});
  try{
    const port=await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('Proof server did not start: '+script)),30000);server.once('message',message=>{clearTimeout(timer);resolve(message.port);});server.once('error',error=>{clearTimeout(timer);reject(error);});server.once('exit',code=>{clearTimeout(timer);reject(Error('Proof server exited: '+script+' ('+code+')'));});});
    return await run(port);
  }finally{server.kill('SIGTERM');}
}
