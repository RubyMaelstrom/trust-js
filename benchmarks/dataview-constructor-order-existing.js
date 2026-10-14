var output=[],n=0,ab=new ArrayBuffer(0),length={valueOf(){n++;return 0;}};
try{new DataView(ab,1,length)}catch(e){output.push(e.name,n)}
var detached=new ArrayBuffer(1);detached.transfer();n=0;
try{new DataView(detached,0,length)}catch(e){output.push(e.name,n)}
console.log(output.join('|'));
