jQuery.noConflict();

jQuery(function($) {

    $(".form_datetime2").datetimepicker({
        format: "dd MM yyyy - hh:ii",
        autoclose: true,
        todayBtn: true,
        startDate: "2013-02-14 10:00",
        minuteStep: 10,
        pickerPosition: "bottom-left"
    });

    $('.chosen').chosen();

    $("#mask_phone").inputmask("mask", {"mask": "+(999) 99-999-999"}); 

    $('.uniform').uniform();

});