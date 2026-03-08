function drawChart(fake, real){

const ctx = document.getElementById('resultChart');

if(window.resultChart){
window.resultChart.destroy();
}

window.resultChart = new Chart(ctx, {

type: 'doughnut',

data: {

labels: ['Fake', 'Real'],

datasets: [{

data: [fake, real],

backgroundColor: [
'#ef4444',
'#22c55e'
],

borderWidth: 0

}]

},

options: {

plugins: {

legend: {

labels: {
color: "black"
}

}

}

}

});

}