const fs=require('fs'),vm=require('vm'),assert=require('assert');
let config, release, polls=0;
const gate=new Promise(r=>release=r);
const core={squadData:{starters:[1],bench:[2]},decisionCard:{captain:1},contingencyPlans:{plan_a:{title:'same snapshot'}}};
async function fetch(path){
 if(path==='/api/advice/start')return {ok:true,json:async()=>({id:'job',status:'calculating',stage:'shared plan'})};
 if(path==='/api/advice/status/job'){polls++;await gate;return {ok:true,json:async()=>({id:'job',status:'ready',result:core})};}
 throw Error(path);
}
vm.runInNewContext(fs.readFileSync('web/static/js/app.js','utf8'),{Vue:{createApp(c){config=c;return {mount(){return {}}}}},window:{},console,setTimeout(fn){return setTimeout(fn,0)},clearTimeout,AbortController,fetch});
(async()=>{
 const obj=config.data();Object.assign(obj,config.methods);obj.$nextTick=async()=>{};
 for(const name of ['loadBasicSquad','loadGameState','loadProfile'])obj[name]=async()=>{};
 for(const name of ['loadHealth','loadContingencyMatrix','loadPriceChanges','loadChecklist','loadSystemStatus','loadChips','loadLeague','loadBriefing'])obj[name]=async()=>new Promise(()=>{});
 const p=obj.refreshAll();await new Promise(r=>setTimeout(r,20));
 assert.deepEqual(obj.squadData,{});assert.equal(obj.decisionCard,null);assert.equal(obj.decisionCardLoading,true);
 release();await p;
 assert.equal(obj.decisionCard.captain,1);assert.equal(obj.squadData.starters[0],1);
 assert.equal(obj.contingencyPlans.plan_a.title,'same snapshot');
 const generation=obj.snapshotGeneration;obj.snapshotGeneration++;
 obj.publishSnapshot('decisionCard',{captain:99},generation);assert.equal(obj.decisionCard.captain,1);
 assert.equal(polls,1);
 console.log('Durable core publication, nonblocking hung auxiliaries, stale-generation guard: passed');
})().catch(e=>{console.error(e);process.exitCode=1});
