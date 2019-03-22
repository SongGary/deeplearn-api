<!DOCTYPE html>
<html lang="en">
<head>
<title>SaaS深度学习平台</title>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<link rel="stylesheet" href="../web/css/bootstrap.min.css" />
<link rel="stylesheet" href="../web/css/matrix-style.css" />
<link rel="stylesheet" href="../web/css/style.css" />
<link rel="stylesheet" href="../web/css/select2.css">

<script src="../web/js/jquery-1.7.1.min.js"></script>
<script type="text/javascript" src="../web/js/select2.min.js"></script>
</head>
<body>
<!--main-container-part-->
<div id="content">
 <div class="container-fluid">
    <div class="row-fluid">
      <div class="span12">
        <!--widget-box-->
                <div id="content-header">
                <h3>预测结果列表：</h3>
                        </div>
                
                        <table class="table table-striped table-bordered table-hover datatable">
            <tbody>	
            	    	<tr>
                                                        <th>实例</th>
                                                        <th>预测结果</th>
                         </tr>
						{%	for key in data.keys(): %}
                            
                        <tr bgcolor="white">
                                <th bgcolor="white">{{key}}</th>
                                <th bgcolor="white">{{data[key]}}</th>
                        </tr>
                        {% endfor %}

             </tbody>
          </div>

        <!--widget-box end-->
        <!--分页-->
        <div class="pagination" id="page_buttom_div">
          <ul>
          </ul>
        </div>
        <!--分页 end-->
      </div>
    </div>
  </div>
</div>
</body>
<!--end-main-container-part-->


</html>