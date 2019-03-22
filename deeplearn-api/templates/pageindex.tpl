<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>深度学习平台</title>
  
    <meta name="description" content="">
    <meta name="author" content="">

    <!-- Le styles -->
    <link href="../web/css/bootstrap.css" rel="stylesheet">
    <link href="../web/css/bootstrap-responsive.css" rel="stylesheet">
    <link href="../web/css/stylesheet.css" rel="stylesheet">
    <link href="../web/css/index.css" rel="stylesheet">
  
    

    <!-- Le fav and touch icons -->
    <link rel="apple-touch-icon-precomposed" sizes="144x144" href="../web/img/apple-touch-icon-144-precomposed.html">
    <link rel="apple-touch-icon-precomposed" sizes="114x114" href="../web/img/apple-touch-icon-114-precomposed.html">
      <link rel="apple-touch-icon-precomposed" sizes="72x72" href="../web/img/apple-touch-icon-72-precomposed.html">
                    <link rel="apple-touch-icon-precomposed" href="../web/img/apple-touch-icon-57-precomposed.html">
                                  

    <!-- HTML5 shim, for IE6-8 support of HTML5 elements -->
    <!--[if lt IE 9]>
      <script src="js/html5shiv.js"></script>
    <![endif]-->
  <!--图标样式-->


<!--主要样式-->
<link rel="stylesheet" type="text/css" href="../web/css/style.css" />

<script type="text/javascript" src="../web/js/jquery-1.7.2.min.js"></script>
<script type="text/javascript">
$(function(){
    $('.tree li:has(ul)').addClass('parent_li').find(' > span').attr('title', 'Collapse this branch');
    $('.tree li.parent_li > span').on('click', function (e) {
        var children = $(this).parent('li.parent_li').find(' > ul > li');
        if (children.is(":visible")) {
            children.hide('fast');
            $(this).attr('title', 'Expand this branch').find(' > i').addClass('icon-plus-sign').removeClass('icon-minus-sign');
        } else {
            children.show('fast');
            $(this).attr('title', 'Collapse this branch').find(' > i').addClass('icon-minus-sign').removeClass('icon-plus-sign');
        }
        e.stopPropagation();
    });
});
</script>
    <script>
    function getclassname(obj){
		if(document.getElementsByClassName('tab_onclick').length==0){
			obj.className='tab_onclick';
			obj.id='tab_onclick';
			}else{
				var obj1=document.getElementById('tab_onclick');
				obj1.className='111';
				obj1.id='1';
				obj.className='tab_onclick';
			   obj.id='tab_onclick';
				
			
			}

		
		}
    </script>
<style type="text/css">
.tree{ width:300px;
height:500px;
background-color:rgb(238,243,247);
float:left;
overflow-y:scroll;
padding:0px;
border:0px;
border-radius:0px;}
</style>
  </head>

  <body>

    
    <div id="content"> <!-- Content start -->
      <div class="inner_content">
          <div class="widgets_area">
                <div class="row-fluid">
                    <div class="span12">
                         <div  class="daohanglink"style="">
                           <span class="daohang"></span>
                           <span>SaaS深度学习平台</span><span>></span>
                     
                           
                         </div>
                         <div class="well brown" style=" border:0px; padding:0px;">
                         <div class="well-content" style="border:0px; padding:0px; background-color:#FFF; height:550px;">
                               
                               <!--tree begin-->
                         <div class="user_tab">
                         	<ul>
                                <a href="/db_setting" target="table"><li onClick="getclassname(this)" ><span>上传数据</span></li></a>
                                <a href="/model_setting" target="table"><li onClick="getclassname(this)" ><span>构建模型</span></li></a>
                           		<a href="/batch_setting" target="table"><li onClick="getclassname(this)" ><span>批量预测</span></li></a>
                            </ul>
                        </div>
                               
                               <!--tree end-->
                                <!--right begin-->
                                 <div class="user_iframe">
                                   
                                   <iframe src="/db_setting" name="table" frameborder="0" width="100%" style="height:550px;">
                                   
                                   </iframe>
                                        
                                 </div>
                               <!--right end-->
                               
                            </div>
                        </div>
                        
                    </div>
                </div>

            
            </div>
        </div>
    </div>

    <!-- Le javascript
    ================================================== -->
    <!-- Placed at the end of the document so the pages load faster -->
    <script src="../web/js/jquery-1.10.2.js"></script>
    <script src="../web/js/jquery-ui-1.10.3.js"></script>
    <script src="../web/js/bootstrap.js"></script>

    <script src="../web/js/flatpoint_core.js"></script>


  </body>
</html>
