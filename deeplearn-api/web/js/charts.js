jQuery.noConflict();

jQuery(function($) {

  function showTooltip(title, x, y, contents) {
    $('<div id="tooltip" class="chart-tooltip"><div class="date">' + title + '<\/div><div class="percentage">Percent: <span>' + x / 10 + '%<\/span><\/div><div class="visits">Visitors: <span>' + x * 12 + '<\/span><\/div><\/div>').css({
        position: 'absolute',
        display: 'none',
        top: y - 117,
        left: x - 91,
        'background-color': '#fff',
        border: '1px solid #5c5c5c'
    }).appendTo("body").fadeIn(200);
    }

  var d1 = [4.3, 5.1, 4.3, 5.2, 5.4, 4.7, 3.5, 4.1, 5.6, 7.4, 6.9, 7.1,
    7.9, 7.9, 7.5, 6.7, 7.7, 7.7, 7.4, 7.0, 7.1, 5.8, 5.9, 7.4,
    8.2, 8.5, 9.4, 8.1, 10.9, 10.4, 10.9, 12.4, 12.1, 9.5, 7.5,
    7.1, 7.5, 8.1, 6.8, 3.4, 2.1, 1.9, 2.8, 2.9, 1.3, 4.4, 4.2,
    3.0, 3.0], 

  d2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.3, 0.0,
      0.0, 0.4, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.6, 1.2, 1.7, 0.7, 2.9, 4.1, 2.6, 3.7, 3.9, 1.7, 2.3,
      3.0, 3.3, 4.8, 5.0, 4.8, 5.0, 3.2, 2.0, 0.9, 0.4, 0.3, 0.5, 0.4], 

  options = {
    series: {
        lines: { 
            show: true, 
            fill: true, 
            lineWidth: 2, 
            steps: false, 
            fillColor: { colors: [{opacity: 0.25}, {opacity: 0}] } 
        },
        points: { 
            show: true, 
            radius: 4, 
            fill: true,
            lineWidth: 1.5
        }
      }, 
      tooltip: true, 
      tooltipOpts: {
          content: '%s: %y'
      }, 
      xaxis: {  mode: "time" , minTickSize: [1, "hour"]
        }, 
      grid: { borderWidth: 0, hoverable: true },
      legend: {
          show: false
    }
  };

  var dt1 = [], dt2 = [], st = new Date(2009, 9, 6).getTime();

  for( var i = 0; i < d2.length; i++ )
  {
      dt1.push([st + i * 3600000, parseFloat( (d1[i]).toFixed( 3 ) )]);
      dt2.push([st + i * 3600000, parseFloat( (d2[i]).toFixed( 3 ) )]);
  }

  var data = [
      { data: dt1, color: '#d24726', label: 'This month sales', lines: { lineWidth: 1.5 } }, 
      { data: dt2, color: '#ac193d', label: 'Last month profit', points: { show: false }, lines: { lineWidth: 2, fill: false } }
  ];

  $.plot($("#chartLine01"), data, options);

  var previousPoint = null;
  $("#chartLine01").bind("plothover", function (event, pos, item) {
    $("#x").text(pos.x.toFixed(2));
    $("#y").text(pos.y.toFixed(2));
    if (item) {
        if (previousPoint != item.dataIndex) {
            previousPoint = item.dataIndex;

            $("#tooltip").remove();
            var x = item.datapoint[0].toFixed(2),
                y = item.datapoint[1].toFixed(2);

                var d = new Date(item.datapoint[0]);

      var monthNames = ["January", "February", "March", "April", "May", "June",  
        "July", "August", "September", "October", "November", "December"];  
        var current_month = d.getMonth();  
        var month_name = monthNames[current_month]; 
      var day = d.getDate();

      var time = (d.getHours()<10?'0':'') + d.getHours() + ":" + (d.getMinutes()<10?'0':'') + d.getMinutes();

      var output = ((''+day).length<2 ? '0' : '') + day + ' ' +
      ((''+month_name).length<2 ? '0' : '') + month_name + ', ' +
      d.getFullYear() + '<span class="clock">' + time + '</span>';

            showTooltip(output, item.pageX, item.pageY, item.series.label + " of " + x + " = " + y);
        }
    } else {
        $("#tooltip").remove();
        previousPoint = null;
    }
  });

  var sin = [], cos = [];
    for (var i = 0; i < 21; i += 0.5) {
      sin.push([i, Math.sin(i)]);
      cos.push([i, Math.cos(i)]);
  }

  var plot = $.plot($("#chartline02"),
    [ { data: sin, label: "Sales (this month)"}, { data: cos, label: "Profit (this month)" } ], {
      series: {
      lines: { show: true },
      points: { show: true, lineWidth: 1.5, radius: 4 }, 
      shadowSize: 0,
    },
      grid: { hoverable: true, clickable: true, borderWidth: 0 },
      yaxis: { min: -1.2, max: 1.2 },
        colors: [ "#ac193d", "#0072c6" ],
          legend: {
              show: false
          }
  });

  var d1 = [[1262304000000, 2043], [1264982400000, 2564], [1267401600000, 2043], [1270080000000, 2198], [1272672000000, 2660], [1275350400000, 2782], [1277942400000, 2430], [1280620800000, 2427], [1283299200000, 2100], [1285891200000, 1214], [1288569600000, 1557], [1291161600000, 2645]];
 
  var data1 = [
      { 
          label: "Sales", 
          data: d1, 
          color: '#0072c6' 
      }
  ];

  $.plot($("#chart01"), data1, {
      xaxis: {
          show: true,
          min: (new Date(2009, 12, 1)).getTime(),
          max: (new Date(2010, 11, 2)).getTime(),
          mode: "time",
          tickSize: [1, "month"],
          monthNames: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
          tickLength: 1,
          axisLabel: 'Month',
          axisLabelFontSizePixels: 11
      },
      yaxis: {
          axisLabel: 'Amount',
          axisLabelUseCanvas: true,
          axisLabelFontSizePixels: 11,
          autoscaleMargin: 0.01,
          axisLabelPadding: 5
      },
      series: {
          lines: {
              show: true, 
              fill: true,
              fillColor: { colors: [ { opacity: 0.5 }, { opacity: 0.2 } ] },
              lineWidth: 1.5
          },
          points: {
              show: true,
              radius: 4,
              fill: true,
              fillColor: "#ffffff",
              symbol: "circle",
              lineWidth: 1.1
          }
      },
     grid: { hoverable: true, clickable: true },
      legend: {
          show: false
      }
  });

  var d1 = [[1262304000000, 2043], [1264982400000, 2564], [1267401600000, 2043], [1270080000000, 2198], [1272672000000, 2660], [1275350400000, 2782], [1277942400000, 2430], [1280620800000, 2427], [1283299200000, 2100], [1285891200000, 1214], [1288569600000, 1557], [1291161600000, 2645]];
 
  var data1 = [
      { 
          label: "Earnings", 
          data: d1, 
          color: '#ac193d' 
      }
  ];

  $.plot($("#chart02"), data1, {
      xaxis: {
          show: true,
          min: (new Date(2009, 12, 1)).getTime(),
          max: (new Date(2010, 11, 2)).getTime(),
          mode: "time",
          tickSize: [1, "month"],
          monthNames: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
          tickLength: 1,
          axisLabel: 'Month',
          axisLabelFontSizePixels: 11
      },
      yaxis: {
          axisLabel: 'Amount',
          axisLabelUseCanvas: true,
          axisLabelFontSizePixels: 11,
          autoscaleMargin: 0.01,
          axisLabelPadding: 5
      },
      series: {
          lines: {
              show: true, 
              fill: true,
              fillColor: { colors: [ { opacity: 0.5 }, { opacity: 0.2 } ] },
              lineWidth: 1.5
          },
          points: {
              show: true,
              radius: 4,
              fill: true,
              fillColor: "#ffffff",
              symbol: "circle",
              lineWidth: 1.1
          }
      },
     grid: { hoverable: true, clickable: true },
      legend: {
          show: false
      }
  });

  var d1 = [[1262304000000, 2043], [1264982400000, 2564], [1267401600000, 2043], [1270080000000, 2198], [1272672000000, 2660], [1275350400000, 2782], [1277942400000, 2430], [1280620800000, 2427], [1283299200000, 2100], [1285891200000, 1214], [1288569600000, 1557], [1291161600000, 2645]];
 
  var data1 = [
      { 
          label: "Visitors", 
          data: d1, 
          color: '#008299' 
      }
  ];

  $.plot($("#chart03"), data1, {
      xaxis: {
          show: true,
          min: (new Date(2009, 12, 1)).getTime(),
          max: (new Date(2010, 11, 2)).getTime(),
          mode: "time",
          tickSize: [1, "month"],
          monthNames: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
          tickLength: 1,
          axisLabel: 'Month',
          axisLabelFontSizePixels: 11
      },
      yaxis: {
          axisLabel: 'Amount',
          axisLabelUseCanvas: true,
          axisLabelFontSizePixels: 11,
          autoscaleMargin: 0.01,
          axisLabelPadding: 5
      },
      series: {
          lines: {
              show: true, 
              fill: true,
              fillColor: { colors: [ { opacity: 0.5 }, { opacity: 0.2 } ] },
              lineWidth: 1.5
          },
          points: {
              show: true,
              radius: 4,
              fill: true,
              fillColor: "#ffffff",
              symbol: "circle",
              lineWidth: 1.1
          }
      },
     grid: { hoverable: true, clickable: true },
      legend: {
          show: false
      }
  });

  var data = [],
    series = Math.floor(Math.random() * 6) + 3;

  for (var i = 0; i < series; i++) {
    data[i] = {
      label: "Series" + (i + 1),
      data: Math.floor(Math.random() * 100) + 1
    }
  }

  var placeholder = $("#piechart");

  $.plot(placeholder, data, {
    series: {
      pie: { 
        show: true,
        radius: 1,
        label: {
          show: true,
          radius: 2/3,
          formatter: function(label, series){
            return '<div style="font-size:8pt;text-align:center;padding:2px;color:white;">'+label+'<br/>'+Math.round(series.percent)+'%</div>';
          },
          threshold: 0.1
        }
      }
    },
    legend: {
      show: false
    }
  });

  var data = [],
    series = Math.floor(Math.random() * 6) + 3;

  for (var i = 0; i < series; i++) {
    data[i] = {
      label: "Series" + (i + 1),
      data: Math.floor(Math.random() * 100) + 1
    }
  }

  var placeholder2 = $("#donutchart");

  $.plot(placeholder2, data, {
    series: {
      pie: { 
        show: true,
        innerRadius: 0.5,
        radius: 1,
        label: {
          show: false,
          radius: 2/3,
          formatter: function(label, series){
            return '<div style="font-size:11px;text-align:center;padding:4px;color:white;">'+label+'<br/>'+Math.round(series.percent)+'%</div>';
          },
          threshold: 0.1
        }
      }
    },
    legend: {
      show: true,
      noColumns: 1, // number of colums in legend table
      labelFormatter: null, // fn: string -> string
      labelBoxBorderColor: "#000", // border color for the little label boxes
      container: null, // container (as jQuery object) to put legend in, null means default on top of graph
      position: "ne", // position of default legend container within plot
      margin: [5, 10], // distance from grid edge to default legend container within plot
      backgroundColor: "#efefef", // null means auto-detect
      backgroundOpacity: 1 // set to 0 to avoid background
    },
    grid: {
      hoverable: true,
      clickable: true
    },
  });

  var data = [],
    series = Math.floor(Math.random() * 6) + 3;

  for (var i = 0; i < series; i++) {
    data[i] = {
      label: "Series" + (i + 1),
      data: Math.floor(Math.random() * 100) + 1
    }
  }

  var placeholder3 = $("#rectangularchart");

  $.plot(placeholder3, data, {
    series: {
      pie: { 
        show: true,
        radius: 500,
        label: {
          show: true,
          radius: 2/3,
          formatter: function(label, series){
            return '<div style="font-size:8pt;text-align:center;padding:2px;color:white;">'+label+'<br/>'+Math.round(series.percent)+'%</div>';
          },
          threshold: 0.1
        }
      }
    },
    legend: {
      show: false
    }
  });

  var previousPoint;
 
  var d1 = [];
  for (var i = 0; i <= 3; i += 1)
      d1.push([i, parseInt(Math.random() * 30)]);

  var d2 = [];
  for (var i = 0; i <= 3; i += 1)
      d2.push([i, parseInt(Math.random() * 30)]);

  var d3 = [];
  for (var i = 0; i <= 3; i += 1)
      d3.push([i, parseInt(Math.random() * 30)]);

  var ds = new Array();

  ds.push({
      data:d1,
      bars: {
          show: true, 
          barWidth: 0.2, 
          order: 1
      },
      color: '#0072c6'
  });
  ds.push({
      data:d2,
      bars: {
          show: true, 
          barWidth: 0.2, 
          order: 2
      },
      color: '#ac193d'
  });
  ds.push({
      data:d3,
      bars: {
          show: true, 
          barWidth: 0.2, 
          order: 3
      },
      color: '#82ba00'
  });
              
  //Display graph
  $.plot($("#barchart"), ds, {
      grid:{ hoverable:true },
  });

  var previousPoint;
 
  var d1_h = [];
  for (var i = 0; i <= 3; i += 1)
      d1_h.push([parseInt(Math.random() * 30),i ]);

  var d2_h = [];
  for (var i = 0; i <= 3; i += 1)
      d2_h.push([parseInt(Math.random() * 30),i ]);

  var d3_h = [];
  for (var i = 0; i <= 3; i += 1)
      d3_h.push([ parseInt(Math.random() * 30),i]);

  var ds_h = new Array();

  ds_h.push({
      data:d1_h,
      bars: {
          show: true, 
          barWidth: 0.2, 
          horizontal: true,
          order: 1
      },
      color: '#0072c6'
  });
  ds_h.push({
      data:d2_h,
      bars: {
          show: true, 
          barWidth: 0.2, 
          horizontal: true,
          order: 2
      },
      color: '#ac193d'
  });
  ds_h.push({
      data:d3_h,
      bars: {
          show: true, 
          barWidth: 0.2, 
          horizontal: true,
          order: 3
      },
      color: '#82ba00'
  });
              
  //Display graph
  $.plot($("#hchart"), ds_h, {
      grid:{ hoverable:true },
  });

});