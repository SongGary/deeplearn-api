/* Dont delete this code!!! This file has to be included in template or template will lose his main functionality */
/* This is main core javascript for properly functioning template - DO NOT DELETE OR EXCLUDE THIS FILE */

jQuery.noConflict();

/** Debounced resize - resizing screen update width and height of some elements **/
(function($,sr){
  var debounce = function (func, threshold, execAsap) {
    var timeout;

    return function debounced () {
        var obj = this, args = arguments;
        function delayed () {
            if (!execAsap)
                func.apply(obj, args);
            timeout = null; 
        };

        if (timeout)
            clearTimeout(timeout);
        else if (execAsap)
            func.apply(obj, args);

        timeout = setTimeout(delayed, threshold || 100); 
    };
  }
  // smartresize 
  jQuery.fn[sr] = function(fn){  return fn ? this.bind('resize', debounce(fn)) : this.trigger(sr); };
})(jQuery,'smartresize');

/** Functions start here **/

jQuery(function($) {

  /* Responsive menu show/hide */
  $('.responsive_menu a').click(function() {
    if($('#main_navigation > div').length > 0) {
      if($('#main_navigation > div').is(":hidden")) {
        $('#main_navigation > div').slideDown();
      } else {
        $('#main_navigation > div').slideUp();
      }
      return false;
    }

    if($('#top_navigation').length > 0) {
      if($('#top_navigation').is(':hidden')) {
        $('#top_navigation').slideDown();
      } else {
        $('#top_navigation').slideUp();
      }
      return false;
    }
  });

  /** Popovers **/
  $('a[rel="popover"]').popover();
  
  $('a[rel="popover"]').click(function() {
    return false;
  });

  /** Tooltips **/
  $("[rel='tooltip']").tooltip();

  //* If main navigation doesn't exist eet content margin-left to 0 *//
  function mainNav() {
    if($('#main_navigation').length === 0) {
      $('#content').css('margin-left', '0px')
    }
  }
  mainNav();

  //* Main menu functions *//
  $('ul.main > li').each(function() {
    var sub_main = $(this).find('ul');
    if (sub_main.length > 0) {
      $(this).children('a').addClass('expand');
      $(this).children('a').append('<span class="count"><i class="icon-chevron-down"></i></span>');
    }
  });

  $('.expand').collapsible({
    defaultOpen: 'current,third',
    cookieName: 'navAct',
    cssOpen: 'subOpened',
    cssClose: 'subClosed',
    speed: 200
  });

  //* Inner navigation height - margin from top *//
  function nav_positions() {
    if($(window).width() > 768) {
      var windowHeight = $(window).height();
      if($('header').css('position') === 'fixed') {
        $('.inner_navigation').css('height', windowHeight - 45 + 'px');
        $('#content').css('padding-top', '45px')
      } else {
        $('#main_navigation').css('position', 'absolute');
        if ($('body').scrollTop() > 46) {
          $('#main_navigation').css({
            'position': 'fixed'
          });
          $('.inner_navigation').css({
            'margin-top': '0',
            'height': windowHeight - 1 + 'px'
          });
          $(".inner_navigation").mCustomScrollbar('update');
        } else {
          $('#main_navigation').css({
            'position': 'absolute'
          });
          $('.inner_navigation').css({
            'margin-top': '75px',
            'height': windowHeight - 78 + 'px'
          });
		  $('.inner_navigation1').css({
            
            'height': windowHeight - 88 + 'px'
          });
		
          $(".inner_navigation").mCustomScrollbar('update');
        }
      }
    } else {
      $('.inner_navigation').css('height', 'inherit');
    }
  }
  nav_positions();

  $(window).scroll(function(){
    nav_positions();
  });

  //* Inner navigation scrolling if go over height *//
  function customScrollMainNavigation() {
    if($(window).width() > 767) {
      if($(".inner_navigation").hasClass('mCustomScrollbar')) {
        return false
      } else {
        $(".inner_navigation").mCustomScrollbar({
          advanced:{
            updateOnContentResize: true,
            updateOnBrowserResize: true
          }
        });
      }
    } else {
      $(".inner_navigation").mCustomScrollbar("destroy");
    }
  }
  customScrollMainNavigation();

  function updateScrollbar() {
    $('.inner_navigation').mCustomScrollbar("update");
  }

  $(window).smartresize(function(){
    customScrollMainNavigation();
    nav_positions();
    updateScrollbar();
  });

  //* Message system min height *//
  function messageheight() {
    var minHeight = $('.message_center .tab-list').height() -1;
    $('.message_center .message_list').css('min-height', minHeight + 'px')
  }
  messageheight();

  //* Footer width calculation based on main navigation *//
  function footermargin() {
    if($('#main_navigation').is(":hidden")) {
      $('footer').css('margin-left', '0px')
    } else {
      $('footer').css('margin-left', '220px')
    }
  }
  footermargin();

  $(window).smartresize(function(){
    footermargin();
  });

  //* Images hover effects *//
  $('.view').hover(function() {
    if($(this).hasClass('view-options')) {
      $(this).find('img').animate({
        'margin-left': '-50%'
      });
      $(this).find('.overlay').animate({
        'width': '50%',
        'opacity': '0.5'
      });
    } else {
      $(this).find('.overlay').animate({
        'opacity': '0.5'
      });
    }
  }, function() {
    if($(this).hasClass('view-options')) {
      $(this).find('img').animate({
        'margin-left': '0'
      });
      $(this).find('.overlay').animate({
        'width': '100%',
        'opacity': '0'
      });
    } else {
      $(this).find('.overlay').animate({
        'opacity': '0'
      });
    }   
  });

  //* Header menu dropdown *//
  $('.header_actions > li > a').click(function() {
    $('.header_actions > li > ul').fadeOut(50);
    var sub = $(this).parent().find('ul');
    if(sub.length > 0) {
      if(sub.is(":hidden")) {
        sub.fadeIn(50);
      } else {
        sub.fadeOut(50);
      }
      return false;
    }
  });

  $('.header_actions > li').click(function(event) {
    event.stopPropagation();
  });

  $(document).click(function() {
    $('.header_actions > li > ul').fadeOut(50);
  });

  //* Show hide main navigation *//
  function mainNavHidden() {
    if($('#main_navigation').is(":hidden")) {
      $('#content').css('margin-left', '0px');
    } else {
      $('#content').css('margin-left', '219px');
    }
  }

  $('.header_actions li a.hide_navigation').click(function() {
    var main_navigation = $('#main_navigation');
    if(main_navigation.is(":hidden")) {
      main_navigation.css('display','block');
      mainNavHidden();
      $(this).find('i').removeClass('icon-chevron-right');
      $(this).find('i').addClass('icon-chevron-left');
    } else {
      main_navigation.css('display','none');
      mainNavHidden();
      $(this).find('i').removeClass('icon-chevron-left');
      $(this).find('i').addClass('icon-chevron-right');
    }
    footermargin();
    return false;
  });

  //* Draggable boxes UI *//
  $("#sortable_boxes").sortable({
    connectWith: ".well",
    items: ".well",
    opacity: 0.8,
    coneHelperSize: true,
    placeholder: 'sortable-box-placeholder round-all',
    forcePlaceholderSize: true,
    tolerance: "pointer"
  });

  $(".column").disableSelection();

  //* Collapsible wells
  $('li.collapse_well a').click(function() {
    var container = $(this).parents('.well').find('.well-content');
    if(container.is(":hidden")) {
      container.slideDown();
      $(this).find('i').removeClass('icon-plus').addClass('icon-minus');
    } else {
      container.slideUp();
      $(this).find('i').removeClass('icon-minus').addClass('icon-plus')
    }
    return false;
  });

  //* Spark lines on dashboard *//
  $("#sparkline").sparkline([5,6,7,5,1,5,2,5,2,3,7,1,2,4,7], {
      type: 'bar',
      height: '30',
      barSpacing: 2,
      barColor: '#0072c6',
      negBarColor: '#ac193d'
  });

  $("#sparkline2").sparkline([3,5,7,2,4,6,7,1,3,6,2,4,6,5,2], {
      type: 'bar',
      height: '30',
      barSpacing: 2,
      barColor: '#ac193d',
      negBarColor: '#ac193d'
  });







  //* You can remove code below if you want to pick colors manually and after you set your template
  //* This is junk code - used for live preview setting/changing live colors on wells, header, main navigation...

  /* Well color schemes */
  $('.well-header > ul > li > a').click(function() {
    var dropdown = $(this).parent('li').children('ul');
    $('.well-header > ul > li > ul').fadeOut(50);
    if(dropdown.is(":hidden")) {
      dropdown.show();
    } else {
      dropdown.hide();
    }
    return false;
  });

  $('a.set_color').click(function() {
    var element = $(this).attr('class').split(' ')[0]
    $(this).parents('.well').removeClass().addClass('well ' + element)
    return false;
  });

  $('.header_actions a.set_color').click(function() {
    var element = $(this).attr('class').split(' ')[0]
    $(this).parents('header').removeClass().addClass(element)
    return false;
  });

  $('li.navigation_color_pick > ul > li > a').click(function() {
    var element = $(this).attr('class').split(' ')[0]
    $('#main_navigation').removeClass().addClass(element)
    return false;
  });

  $('a.dark_navigation').click(function() {
    var element = $(this).attr('class').split(' ')[0]
    $('#main_navigation').removeClass().addClass(element)
    return false;
  });

});