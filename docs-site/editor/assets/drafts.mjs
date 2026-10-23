export function draftStore(user,workspace,storage=localStorage){
  const prefix=`limen-editor:${user}:${workspace}:`,key=patch=>JSON.stringify([patch.path,patch.baseBlobSha,patch.sourceRange,patch.originalText]);
  function records(){const items=[];for(let index=0;index<storage.length;index++){const name=storage.key(index);if(name.startsWith(prefix+'draft:'))items.push(JSON.parse(storage.getItem(name)));}return items;}
  function heads(){const items=records(),parents=new Set(items.flatMap(item=>item.previous));return items.filter(item=>!parents.has(item.id)&&!item.discard&&!storage.getItem(prefix+'ack:'+item.id));}
  function save(patch,previous=[]){if(storage.getItem(`limen-editor:${user}:signed-out`))throw Error('This writer signed out; sign in again before saving prose');const id=crypto.randomUUID(),item={id,patch:{...patch,draftId:id},previous};storage.setItem(prefix+'draft:'+id,JSON.stringify(item));return item;}
  function undo(id){const item=heads().find(item=>item.id===id);if(!item)return;const record={...item,id:crypto.randomUUID(),previous:[id],discard:true};storage.setItem(prefix+'draft:'+record.id,JSON.stringify(record));}
  function confirmed(ids){for(const id of ids)storage.setItem(prefix+'ack:'+id,'committed');}
  function conflicts(){const grouped=new Map();for(const item of heads()){const group=key(item.patch);grouped.set(group,(grouped.get(group)||0)+1);}return [...grouped.values()].some(count=>count>1);}
  return {prefix,heads,save,undo,confirmed,conflicts};
}

export function writerStores(user,storage=localStorage){
  const prefix=`limen-editor:${user}:`,workspaces=new Set();
  for(let index=0;index<storage.length;index++){const name=storage.key(index);if(name.startsWith(prefix)&&name.includes(':draft:'))workspaces.add(name.slice(prefix.length).split(':')[0]);}
  return [...workspaces].map(workspace=>({workspace,store:draftStore(user,workspace,storage)}));
}
export function clearWriter(user,storage=localStorage){const prefix=`limen-editor:${user}:`,names=[];for(let index=0;index<storage.length;index++)if(storage.key(index).startsWith(prefix))names.push(storage.key(index));for(const name of names)storage.removeItem(name);storage.setItem(prefix+'signed-out',crypto.randomUUID());}
