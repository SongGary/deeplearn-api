
displayDataTable();

function displayDataTable() {

	var spobj = document.getElementById("select_host_name");
	var sptext = spobj.options[spobj.selectedIndex].value;
	
	jQuery(function(){
		jQuery.ajax({
			type:"POST",
			url:"/imweb/hlpage/table/getdata",
			data:{host:sptext},
			dataType:"text",
			success: function(jsonStr, textStatus) {
				var obj = JSON.parse(jsonStr);
				var data_context = "";
				for (var i = 0; i < obj.data.length; i++) {
					 data_context += "<tr class=\"odd gradeX\"> "
					var host = "";
					var role = "";
					 for (var j = 0; j < obj.data[i].length; j++) {
						var dtext = obj.data[i][j];
						if (j == 0) {
							host = obj.data[i][j];
						} else if (j == 2) {
							role = obj.data[i][j];
						} else if (j == 3) {
							if (dtext == "alive") {
								dtext = "<font color=\"green\">" + obj.data[i][j] + "</font>"
							} else {
								dtext = "<font color=\"red\">" + obj.data[i][j] + "</font>"
							}
						} else if (j == 5) {
					 		dtext = "<a href=\"#\" class=\"btn btn-mini btn-op\" " +
					 				"onClick=\"nodeRestart(this.id)\" " +
					 				"id=" + host +"#"+role + "^restart " + 
					 				"style=\"font-size: 13.5px;padding: 3px;border-radius: 3px;\">重启</a>"
					 	} else if (j == 6) {
					 		dtext = "<a href=\"#\" class=\"btn btn-mini btn-op\" " +
			 						"onClick=\"nodeStop(this.id)\" " +
			 						"id=" + host +"#"+role + "^stop " +
			 						"style=\"font-size: 13.5px;padding: 3px;border-radius: 3px;\">停止</a>"
					 	} else if (j == 8) {
					 	}
					 	data_context += "<td>" + dtext + " </td>";
					 }
					 data_context += " </tr>";
				}
							
				$("#data_table_div table tbody").html(data_context);
				
				if (obj.data.length == 0) {
					alert("没有查询到数据！")
				}
			}
		});
	});
}

function restartCluster() {
	var r = confirm("你将重新启动整个集群，请确认该操作的必要性！！！");
	if (r == true) {
		doOperation("all", "start");
	}
}

function nodeRestart(nid) {
	var v = nid.split("^");
	var r = confirm("你将重新启动节点：" + v[0] +"，请确认该操作的必要性！！！");
	if (r == true) {
		doOperation(v[0], "start");
	}
}

function nodeStop(nid) {
	var v = nid.split("^");
	var r = confirm("你将停止节点：" + v[0] +"，请确认该操作的必要性！！！");
	if (r == true) {
		doOperation(v[0], "stop");
	}
}

function doOperation(node, op) {
	jQuery(function(){
		jQuery.ajax({
			type:"POST",
			url:"/imweb/hlpage/table/nodeoperation",
			data:{node:node, op:op},
			dataType:"text",
			success: function(jsonStr, textStatus) {
				displayDataTable();
				alert("操作完成！提示信息如下：\n" + jsonStr);
			}
		})
	})
}

jQuery(function(){
	$("#select_host_name").change(function() {
		displayDataTable();
	});
});
