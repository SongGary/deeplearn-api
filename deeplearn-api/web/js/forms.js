jQuery.noConflict();

jQuery(function($) {

  $('#colorpicker').minicolors({
    textfield: false,
  });
  
  $('#tags').tagsInput();

  $('#autosize').autosize();   

  $('.chosen').chosen();

  $("#textare_char").charCount({
    allowed: 150,    
    warning: 20,
    counterText: 'Characters left: '  
  });

  // Dual Box Select
  var db = jQuery('#dualselect').find('.select_arrows button');  //get arrows of dual select
  var sel1 = jQuery('#dualselect select:first-child');    //get first select element
  var sel2 = jQuery('#dualselect select:last-child');     //get second select element
  
  sel2.empty(); //empty it first from dom.
  
  db.click(function(){
    var t = (jQuery(this).hasClass('ds_prev'))? 0 : 1;  // 0 if arrow prev otherwise arrow next
    if(t) {
      sel1.find('option').each(function(){
        if(jQuery(this).is(':selected')) {
          jQuery(this).attr('selected',false);
          var op = sel2.find('option:first-child');
          sel2.append(jQuery(this));
        }
      }); 
    } else {
      sel2.find('option').each(function(){
        if(jQuery(this).is(':selected')) {
          jQuery(this).attr('selected',false);
          sel1.append(jQuery(this));
        }
      });   
    }
    return false;
  }); 

  $('.uniform').uniform();

});