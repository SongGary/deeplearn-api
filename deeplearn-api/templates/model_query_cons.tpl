<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<script language="JavaScript" src="../web/js/mydate.js"></script>
<html>
    <head>
    	<title>深度学习平台</title>
    	<meta HTTP-EQUIV="content-type" CONTENT="text/html; charset=UTF-8">
    	
    	<link rel="stylesheet" type="text/css" href="../web/css/login.css" />
		<script src="../web/js/cufon-yui.js" type="text/javascript"></script>
    </head>
    </head>
    <body onload="load()">
			
<div class="wrapper">
			<div class="content">
				<div id="form_wrapper" >
    				<form class="login active"  action="setmodels" method="POST" enctype="multipart/form-data">
    				<h3>设置构建模型条件：</h3>
    				<tr>
<div class="form_list"><label class="lable_title">设定模型名称：</label><input name="modelname" type="text" value=""></div></tr>
<tr>
<div class="form_list"><label class="lable_title">数据集名：</label><input name="dataset" type="text" value=""></div></tr>
<tr>
<div class="form_list">  <label class="lable_title">算法选项：</label>  				
        <select class='prov' id='prov' name='prov' onchange='changeCity()'>
            <option value='0'>请选择算法</option>
        </select></div></tr>
<tr>
<div id="div1" style="display:none;" class="form_list"><label class="lable_title">另数据集名：</label><input name="dataset02" type="text" value=""></div></tr>                    		
<input type="submit" value="提交" />

						

					</div>
					
    				</form>
    				</div>
				<div class="clear"></div>
			</div>		
		</div>
		<script>
        var province=document.getElementById("prov");
        var arr_prov=new Array(new Option("请选择算法",'default'),new Option("textclassification|文本分类",'textclassification'),new Option("semanticsim|文本相似度","semanticsim"),new Option("textsum|内容摘要","textsum"));
        
        //动态载入所有省份
        function load(){
            for(var i=0;i<arr_prov.length;i++){
                province.options[i]=arr_prov[i];
            }
        }
        //选中省份之后，根据索引动态载入相应城市
        function changeCity(){
            //清空上次的选项
            var index=province.selectedIndex 
            var div1=document.getElementById("div1");  
            if(province.options[index].value=='textsum') 
              div1.style.display='block'; 
            else div1.style.display='none';
        }
    </script>
    </body>
</html>

