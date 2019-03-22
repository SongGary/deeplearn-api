jQuery.noConflict();

jQuery(function($) {
  
  var date = new Date();
  var d = date.getDate();
  var m = date.getMonth();
  var y = date.getFullYear();
  $('#calendar').fullCalendar({
    header: {
    left: 'title',
    center: '',
    right: 'prev,next,today month,agendaWeek,agendaDay'
    },
    buttonText: {
    prev: '<i class="icon-chevron-left cal_prev" />',
    next: '<i class="icon-chevron-right cal_next" />'
    },
    aspectRatio: 1.5,
    selectable: false,
    selectHelper: true,
    editable: true,
    droppable: true,
    theme: false,
    editable: true,
    droppable: true, // this allows things to be dropped onto the calendar !!!
    drop: function(date, allDay) { // this function is called when something is dropped
    
      // retrieve the dropped element's stored Event Object
      var originalEventObject = $(this).data('eventObject');
      
      // we need to copy it, so that multiple events don't have a reference to the same object
      var copiedEventObject = $.extend({}, originalEventObject);
      
      // assign it the date that was reported
      copiedEventObject.start = date;
      copiedEventObject.allDay = allDay;
      
      // render the event on the calendar
      // the last `true` argument determines if the event "sticks" (http://arshaw.com/fullcalendar/docs/event_rendering/renderEvent/)
      $('#calendar').fullCalendar('renderEvent', copiedEventObject, true);
      
      // is the "remove after drop" checkbox checked?
      if ($('#drop-remove').is(':checked')) {
        // if so, remove the element from the "Draggable Events" list
        $(this).remove();
      }
      
    },
    events: [
    {
      title: 'All Day Event',
      start: new Date(y, m, 1)
    },
    {
      title: 'Long Event',
      start: new Date(y, m, d-5),
      end: new Date(y, m, d-2)
    },
    {
      id: 999,
      title: 'Repeating Event',
      start: new Date(y, m, d+8, 16, 0),
      end: new Date(y, m, d+9, 16, 0),
      allDay: false
    },
    {
      title: 'Meeting',
      start: new Date(y, m, d+12, 15, 0),
      allDay: false
    },
    {
    title: 'Doomsday',
            start: new Date(y, m, d+1, 19, 0),
            end: new Date(y, m, d+1, 22, 30),
            allDay: false
          },
    {
    title: 'We are alive',
          start: new Date(y, m, d+2, 19, 0),
          end: new Date(y, m, d+4, 22, 30),
          allDay: false
        },
        {
          title: 'Click for Google',
          start: new Date(y, m, 28),
          end: new Date(y, m, 29),
          url: '../../google.com/default.htm'
        }
      ],
    eventColor: '#5db2ff'
  });

});