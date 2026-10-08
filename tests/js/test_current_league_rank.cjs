const fs=require('fs'),vm=require('vm'),assert=require('assert');let config;
vm.runInNewContext(fs.readFileSync('web/static/js/app.js','utf8'),{Vue:{createApp(c){config=c;return{mount(){return{}}}}},window:{},console});
const o=config.data();o.leagueData=JSON.parse(fs.readFileSync('tests/fixtures/league_current_rank.json','utf8'));assert.equal(o.leagueData.simulation.user_rank,undefined);
const label=()=>config.computed.currentLeagueRankLabel.call(o);
assert.equal(label(),'#7');assert.equal(config.computed.overallRankLabel.call(o),'#123,456');o.aux.league.loading=true;assert.equal(label(),'Loading league rank...');o.aux.league.loading=false;o.aux.league.error='failed';assert.equal(label(),'League rank unavailable');o.aux.league.error=null;o.leagueData={league_rank_status:'not_listed'};assert.equal(label(),'Not listed in this league');console.log('Current league rank uses official top-level field with real simulation contract and loading/error/missing states: passed');
