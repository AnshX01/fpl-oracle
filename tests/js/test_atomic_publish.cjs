const fs=require('fs'),vm=require('vm'),assert=require('assert');
let config;
vm.runInNewContext(fs.readFileSync('web/static/js/app.js','utf8'),{Vue:{createApp(c){config=c;return {mount(){return {}}}}},window:{},console,setTimeout,clearTimeout,AbortController});
async function run(revisions){
 const obj=config.data();Object.assign(obj,config.methods); obj.$nextTick=async()=>{};
 for(const name of ['loadBasicSquad','loadGameState','loadProfile','loadHealth','loadContingencyMatrix','loadPriceChanges','loadChecklist','loadSystemStatus'])obj[name]=async()=>{};
 let release;const gate=new Promise(r=>release=r);
 obj.loadSquad=async()=>{obj.servedRevisions.push(revisions[0]);obj.publishSnapshot('squadData',{starters:[1],bench:[2]});};
 obj.loadDecisionCard=async()=>{await gate;obj.servedRevisions.push(revisions[1]);obj.publishSnapshot('decisionCard',{captain:1});};
 for(const name of ['loadContingencyPlans','loadChips','loadLeague','loadBriefing'])obj[name]=async()=>{};
 const p=obj.refreshAll();await new Promise(r=>setTimeout(r,0));
 assert.deepEqual(obj.squadData,{});assert.equal(obj.decisionCard,null);
 release();await p;
 if(revisions[0]===revisions[1])assert.equal(obj.squadData.starters[0],1);
 else {assert.deepEqual(obj.squadData,{});assert.equal(obj.decisionCard,null);}
}

async function timeoutAndAuxiliary(){
 const obj=config.data();Object.assign(obj,config.methods);obj.$nextTick=async()=>{};
 for(const name of ['loadBasicSquad','loadGameState','loadProfile','loadContingencyPlans','loadChips','loadLeague','loadBriefing'])obj[name]=async()=>{};
 for(const name of ['loadHealth','loadContingencyMatrix','loadPriceChanges','loadChecklist','loadSystemStatus'])obj[name]=async()=>new Promise(()=>{});
 obj.loadSquad=async()=>obj.publishSnapshot('squadData',{starters:[1],bench:[]});
 obj.loadDecisionCard=async()=>obj.publishSnapshot('decisionCard',{captain:1});
 await obj.refreshAll();assert.equal(obj.decisionCard.captain,1); // hung auxiliary did not hold core
 const generation=obj.snapshotGeneration;
 obj.snapshotGeneration++;
 obj.snapshotBuffer={};
 obj.publishSnapshot('decisionCard',{captain:99},generation);
 assert.deepEqual(obj.snapshotBuffer,{}); // old json completion cannot pollute next refresh
 let timeoutCallback;
 vm.runInNewContext(fs.readFileSync('web/static/js/app.js','utf8'),{Vue:{createApp(c){config=c;return {mount(){return {}}}}},window:{},console,
 setTimeout(fn,ms){if(ms===90000){timeoutCallback=fn;return 1;}return setTimeout(fn,ms)},clearTimeout(){},AbortController});
 const timed=config.data();Object.assign(timed,config.methods);timed.$nextTick=async()=>{};
 for(const name of ['loadBasicSquad','loadGameState','loadProfile','loadHealth','loadContingencyMatrix','loadPriceChanges','loadChecklist','loadSystemStatus','loadSquad','loadContingencyPlans','loadChips','loadLeague','loadBriefing'])timed[name]=async()=>{};
 let late;timed.loadDecisionCard=async()=>{const gen=timed.snapshotGeneration;await new Promise(r=>late=r);timed.publishSnapshot('decisionCard',{captain:99},gen)};
 const p=timed.refreshAll();await new Promise(r=>setTimeout(r,0));timeoutCallback();await p;
 assert.equal(timed.decisionCard,null);assert(timed.decisionCardError.includes('90 seconds'));
 timed.loadDecisionCard=async()=>timed.publishSnapshot('decisionCard',{captain:1});
 await timed.refreshAll();late();await new Promise(r=>setTimeout(r,0));assert.equal(timed.decisionCard.captain,1);
 console.log('UI timeout, late generation and nonblocking auxiliary: passed');
}
(async()=>{await run(['1','1']);await run(['1','2']);console.log('UI atomic publish: 2 scenarios passed');await timeoutAndAuxiliary()})().catch(e=>{console.error(e);process.exitCode=1});
