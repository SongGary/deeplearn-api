jQuery.noConflict();

jQuery(function($) {

    $('#rootwizard').bootstrapWizard({
        'nextSelector': '.next-button',
        'previousSelector': '.previous-button',
        onTabClick: function (tab, navigation, index) {
            return false;
        },
        onNext: function(tab, navigation, index) {
            var total = navigation.find('li').length;
            var current = index + 1;

            if (current == 1) {
                $('#rootwizard').find('.previous-button').hide();
            } else {
                $('#rootwizard').find('.previous-button').show();
            }

            if (current >= total) {
                $('#rootwizard').find('.next-button').hide();
                $('#rootwizard').find('.submit-button').show();
            } else {
                $('#rootwizard').find('.next-button').show();
                $('#rootwizard').find('.submit-button').hide();
            }
        },
        onPrevious: function (tab, navigation, index) {
            var total = navigation.find('li').length;
            var current = index + 1;

            if (current == 1) {
                $('#rootwizard').find('.previous-button').hide();
            } else {
                $('#rootwizard').find('.previous-button').show();
            }

            if (current >= total) {
                $('#rootwizard').find('.next-button').hide();
                $('#rootwizard').find('.submit-button').show();
            } else {
                $('#rootwizard').find('.next-button').show();
                $('#rootwizard').find('.submit-button').hide();
            }
        },
        onTabShow: function(tab, navigation, index) {
            console.log('onTabShow');
            var $total = navigation.find('li').length;
            var $current = index+1;
            var $percent = ($current/$total) * 100;
            $('#rootwizard').find('.bar').animate({
                'width': $percent+'%'
            });
        }
    });

    $('#rootwizard').find('.previous-button').hide();
    $('#rootwizard').find('.submit-button').hide();

    var $validator = $("#rootwizard_form").validate({
        rules: {
            name: {
              required: true,
              minlength: 3
            },
            lastname: {
              required: true,
              minlength: 3
            },
            username: {
              required: true,
              minlength: 3,
            },
            password: {
              required: true,
              minlength: 3,
            },
            conpassword: {
              minlength: 3,
              equalTo: "#password"
            },
            email: {
              required: true,
              email: true,
              minlength: 3,
            }
        }
    });

    $('#rootwizard2').bootstrapWizard({
        'nextSelector': '.next-button',
        'previousSelector': '.previous-button',
        onTabClick: function (tab, navigation, index) {
            return false;
        },
        onNext: function(tab, navigation, index) {
            var $valid = $("#rootwizard_form").valid();
            if(!$valid) {
                $validator.focusInvalid();
                return false;
            }

            var total = navigation.find('li').length;
            var current = index + 1;

            if (current == 1) {
                $('#rootwizard2').find('.previous-button').hide();
            } else {
                $('#rootwizard2').find('.previous-button').show();
            }

            if (current >= total) {
                $('#rootwizard2').find('.next-button').hide();
                $('#rootwizard2').find('.submit-button').show();
            } else {
                $('#rootwizard2').find('.next-button').show();
                $('#rootwizard2').find('.submit-button').hide();
            }
        },
        onPrevious: function (tab, navigation, index) {
            var total = navigation.find('li').length;
            var current = index + 1;

            if (current == 1) {
                $('#rootwizard2').find('.previous-button').hide();
            } else {
                $('#rootwizard2').find('.previous-button').show();
            }

            if (current >= total) {
                $('#rootwizard2').find('.next-button').hide();
                $('#rootwizard2').find('.submit-button').show();
            } else {
                $('#rootwizard2').find('.next-button').show();
                $('#rootwizard2').find('.submit-button').hide();
            }
        },
        onTabShow: function(tab, navigation, index) {
            console.log('onTabShow');
            var $total = navigation.find('li').length;
            var $current = index+1;
            var $percent = ($current/$total) * 100;
            $('#rootwizard2').find('.bar').animate({
                'width': $percent+'%'
            });
        }
    });

    $('#rootwizard2 .submit-button').click(function() {
        var $valid = $("#rootwizard_form").valid();
        if(!$valid) {
            $validator.focusInvalid();
            return false;
        }
    });

    $('#rootwizard2').find('.previous-button').hide();
    $('#rootwizard2').find('.submit-button').hide();

    $('#rootwizard3').bootstrapWizard({
        'nextSelector': '.next-button',
        'previousSelector': '.previous-button',
        onTabClick: function (tab, navigation, index) {
            return false;
        },
        onNext: function(tab, navigation, index) {
            var total = navigation.find('li').length;
            var current = index + 1;

            $('li', $('#rootwizard3')).removeClass("done");
            var li_list = $('#rootwizard3').find('li');
            for (var i = 0; i < index; i++) {
                jQuery(li_list[i]).addClass("done");
            }

            if (current == 1) {
                $('#rootwizard3').find('.previous-button').hide();
            } else {
                $('#rootwizard3').find('.previous-button').show();
            }

            if (current >= total) {
                $('#rootwizard3').find('.next-button').hide();
                $('#rootwizard3').find('.submit-button').show();
            } else {
                $('#rootwizard3').find('.next-button').show();
                $('#rootwizard3').find('.submit-button').hide();
            }
        },
        onPrevious: function (tab, navigation, index) {
            var total = $('#rootwizard3').find('li').length;
            var current = index + 1;

            $('li', $('#rootwizard3')).removeClass("done");
            var li_list = $('#rootwizard3').find('li');
            for (var i = 0; i < index; i++) {
                jQuery(li_list[i]).addClass("done");
            }

            if (current == 1) {
                $('#rootwizard3').find('.previous-button').hide();
            } else {
                $('#rootwizard3').find('.previous-button').show();
            }

            if (current >= total) {
                $('#rootwizard3').find('.next-button').hide();
                $('#rootwizard3').find('.submit-button').show();
            } else {
                $('#rootwizard3').find('.next-button').show();
                $('#rootwizard3').find('.submit-button').hide();
            }
        },
        onTabShow: function(tab, navigation, index) {
            console.log('onTabShow');
            var $total = navigation.find('li').length;
            var $current = index+1;
            var $percent = ($current/$total) * 100;
            $('#rootwizard3').find('.bar').animate({
                'width': $percent+'%'
            });
        }
    });

    $('#rootwizard3').find('.previous-button').hide();
    $('#rootwizard3').find('.submit-button').hide();

    $('#rootwizard4').bootstrapWizard({
        'nextSelector': '.next-button',
        'previousSelector': '.previous-button',
        onTabClick: function (tab, navigation, index) {
            return false;
        },
        onNext: function(tab, navigation, index) {
            var total = navigation.find('li').length;
            var current = index + 1;

            $('li', $('#rootwizard4')).removeClass("done");
            var li_list = $('#rootwizard4').find('li');
            for (var i = 0; i < index; i++) {
                jQuery(li_list[i]).addClass("done");
            }

            if (current == 1) {
                $('#rootwizard4').find('.previous-button').hide();
            } else {
                $('#rootwizard4').find('.previous-button').show();
            }

            if (current >= total) {
                $('#rootwizard4').find('.next-button').hide();
                $('#rootwizard4').find('.submit-button').show();
            } else {
                $('#rootwizard4').find('.next-button').show();
                $('#rootwizard4').find('.submit-button').hide();
            }
        },
        onPrevious: function (tab, navigation, index) {
            var total = $('#rootwizard4').find('li').length;
            var current = index + 1;

            $('li', $('#rootwizard4')).removeClass("done");
            var li_list = $('#rootwizard4').find('li');
            for (var i = 0; i < index; i++) {
                jQuery(li_list[i]).addClass("done");
            }

            if (current == 1) {
                $('#rootwizard4').find('.previous-button').hide();
            } else {
                $('#rootwizard4').find('.previous-button').show();
            }

            if (current >= total) {
                $('#rootwizard4').find('.next-button').hide();
                $('#rootwizard4').find('.submit-button').show();
            } else {
                $('#rootwizard4').find('.next-button').show();
                $('#rootwizard4').find('.submit-button').hide();
            }
        },
        onTabShow: function(tab, navigation, index) {
            console.log('onTabShow');
            var $total = navigation.find('li').length;
            var $current = index+1;
            var $percent = ($current/$total) * 100;
            $('#rootwizard4').find('.bar').animate({
                'width': $percent+'%'
            });
        }
    });

    $('#rootwizard4').find('.previous-button').hide();
    $('#rootwizard4').find('.submit-button').hide();

});