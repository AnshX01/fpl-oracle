const fs=require('fs'),vm=require('vm'),assert=require('assert');let config;
const ctx={Vue:{createApp(c){config=c;return {mount(){return {}}}}},window:{},console,setTimeout,clearTimeout,AbortController,AbortSignal,location:{hash:'#league'},history:{replaceState(){}},fetch:async()=>({ok:false,status:503})};
vm.runInNewContext(fs.readFileSync('web/static/js/app.js','utf8'),ctx);
(async()=>{
 const obj=config.data();Object.assign(obj,config.methods);obj.$nextTick=async()=>{};obj.triggerToast=()=>{};
 obj.restoreRoute();assert.equal(obj.activeTab,'league');
 assert.equal(obj.priceLabel('RISE_IMMINENT'),'Strong rise momentum');assert.equal(obj.humanLabel('BENCH_BOOST'),'Bench Boost');assert.equal(obj.humanLabel('1_TRANSFER'),'1 Transfer');assert.equal(obj.overlapLabel({}),'Not measured');
 await obj.loadPriceChanges();assert.equal(obj.aux.prices.loading,false);assert(obj.aux.prices.error);assert.equal(obj.priceChanges.rises,undefined);
 ctx.fetch=async()=>({ok:true,json:async()=>({rises:[{element:1}],falls:[]})});await obj.retryAux('prices');assert.equal(obj.aux.prices.error,null);assert.equal(obj.priceChanges.rises.length,1);
 let release;ctx.fetch=async()=>{await new Promise(r=>release=r);return {ok:true,json:async()=>({rises:[{element:99}]})}};
 const p=obj.loadPriceChanges();obj.snapshotGeneration++;release();await p;assert.equal(obj.priceChanges.rises[0].element,1);
 obj.matchingInProgress=true;obj.manualSquadText='players';await obj.matchPastedSquad();assert.equal(obj.matchingInProgress,true);
 console.log('Polished labels, route restoration, source failure/retry, superseded-source guard, match double-submit guard: passed');
})().catch(e=>{console.error(e);process.exitCode=1});
