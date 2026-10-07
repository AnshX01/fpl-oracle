const fs=require('fs'),vm=require('vm'),assert=require('assert');
let config;
vm.runInNewContext(fs.readFileSync('web/static/js/app.js','utf8'),{Vue:{createApp(c){config=c;return {mount(){return {}}}}},window:{},console});
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
(async()=>{await run(['1','1']);await run(['1','2']);console.log('UI atomic publish: 2 scenarios passed')})();
