jQuery.noConflict();

jQuery(function($) {

  $(".form_datetime").datetimepicker({
    format: 'yyyy-mm-dd hh:ii',
    pickerPosition: "bottom-left"
  });

  $(".form_datetime2").datetimepicker({
        format: "dd MM yyyy - hh:ii",
        autoclose: true,
        todayBtn: true,
        startDate: "2013-02-14 10:00",
        minuteStep: 10,
        pickerPosition: "bottom-left"
    });

  $(".form_datetime3").datetimepicker({
        format: "dd MM yyyy - HH:ii P",
        showMeridian: true,
        autoclose: true,
        todayBtn: true,
        pickerPosition: "bottom-left"
    });

  $(".form_datetime4").datetimepicker({
        format: "dd MM yyyy - HH:ii P",
        showMeridian: true,
        autoclose: true,
        todayBtn: true,
        pickerPosition: "bottom-right"
    });

  $('#timepicker2').timepicker({
    minuteStep: 1,
    showSeconds: true,
    showMeridian: false,
    defaultTime: false
  });

  $('#timepicker3').timepicker({
    minuteStep: 1,
    showSeconds: true,
    showMeridian: true,
    defaultTime: false
  });

  $('#dp').datepicker();
  $('#dp2').datepicker();
  $('#dpYears').datepicker();
  $('#dpMonths').datepicker();

  // disabling dates
  var nowTemp = new Date();
  var now = new Date(nowTemp.getFullYear(), nowTemp.getMonth(), nowTemp.getDate(), 0, 0, 0, 0);

  var checkin = $('#dpd1').datepicker({
    onRender: function(date) {
      return date.valueOf() < now.valueOf() ? 'disabled' : '';
    }
  }).on('changeDate', function(ev) {
    if (ev.date.valueOf() > checkout.date.valueOf()) {
      var newDate = new Date(ev.date)
      newDate.setDate(newDate.getDate() + 1);
      checkout.setValue(newDate);
    }
    checkin.hide();
    $('#dpd2')[0].focus();
  }).data('datepicker');
  var checkout = $('#dpd2').datepicker({
    onRender: function(date) {
      return date.valueOf() <= checkin.date.valueOf() ? 'disabled' : '';
    }
  }).on('changeDate', function(ev) {
    checkout.hide();
  }).data('datepicker');

  // Alert if end date lower then start
  var startDate = new Date(2012,1,20);
  var endDate = new Date(2012,1,25);
  $('#dp4').datepicker()
    .on('changeDate', function(ev){
      if (ev.date.valueOf() > endDate.valueOf()){
        $('#alert').show().html('<strong>Oh snap!</strong> The start date can not be greater then the end date');
      } else {
        $('#alert').hide();
        startDate = new Date(ev.date);
        $('#startDate').text($('#dp4').data('date'));
      }
      $('#dp4').datepicker('hide');
    });
  $('#dp5').datepicker()
    .on('changeDate', function(ev){
      if (ev.date.valueOf() < startDate.valueOf()){
        $('#alert').show().html('<strong>Oh snap!</strong> The end date can not be less then the start date');
      } else {
        $('#alert').hide();
        endDate = new Date(ev.date);
        $('#endDate').text($('#dp5').data('date'));
      }
      $('#dp5').datepicker('hide');
  });

  $('#masked_date').inputmask("d/m/y");
  $('#masked_date2').inputmask("d/m/y",{ "placeholder": "-" });
  $('#masked_date3').inputmask("d/m/y",{ "placeholder": "dd/mm/yyyy" });
  $("#mask_phone").inputmask("mask", {"mask": "+(999) 99-999-999"}); 
  $("#mask_currency").inputmask('€ 999.999.999,99', { numericInput: true, rightAlignNumerics: false });
  $("#mask_ssn").inputmask("999-99-9999", {placeholder:" ", clearMaskOnLostFocus: true }); //default

});