function monthName(date) {
  const monthList = [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December'
  ]

  return monthList[date.getMonth()];
}

function noOfDays(date) {
  const days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

  const year = date.getFullYear();
  const month = date.getMonth();

  if (((year%400 === 0) || (year%100 !== 0 && year%4 === 0)) && (month === 1)) {
    return 29;
  }

  return days[month];
}

function createCalendar(date = new Date()) {
  const container = document.getElementById('news-calendar').appendChild(document.createElement('div'));
  container.className = 'calendar-container';

  // month-year
  const monthYear = container.appendChild(document.createElement('div'));
  monthYear.className = 'month-year';

  const prevYear = monthYear.appendChild(document.createElement('dev'));
  prevYear.className = 'prev-month';
  prevYear.innerHTML = '<';

  const curYear = monthYear.appendChild(document.createElement('dev'));
  curYear.className = 'cur-month';
  const month = curYear.appendChild(document.createElement('div'));
  month.className = 'month';
  month.innerHTML = monthName(date);
  const year = curYear.appendChild(document.createElement('div'));
  year.className = 'year';
  year.innerHTML = date.getFullYear();

  const nextYear = monthYear.appendChild(document.createElement('dev'));
  nextYear.className = 'next-month';
  nextYear.innerHTML = '>';

  // table
  const table = container.appendChild(document.createElement('table'));
  table.className = 'calendar-table';
  
  weekdays = table.appendChild(document.createElement('tr'));
  weekdays.className = 'weekdays';
  weekdays.appendChild(document.createElement('td')).innerHTML = 'Su';
  weekdays.appendChild(document.createElement('td')).innerHTML = 'Mo';
  weekdays.appendChild(document.createElement('td')).innerHTML = 'Tu';
  weekdays.appendChild(document.createElement('td')).innerHTML = 'We';
  weekdays.appendChild(document.createElement('td')).innerHTML = 'Th';
  weekdays.appendChild(document.createElement('td')).innerHTML = 'Fr';
  weekdays.appendChild(document.createElement('td')).innerHTML = 'Sa';

  let week, day, prevMonDate, nextMonthDate = 1;
  let firstDay = new Date(date.getFullYear(), date.getMonth(), 1).getDay();
  let sum = noOfDays(date) + firstDay;
  let noOfWeeks = Math.floor(sum/7);
  if (sum%7 !== 0) {
    noOfWeeks++;
  }

  if (date.getMonth() === 0) {
    prevMonDate = noOfDays(new Date(date.getFullYear() - 1, 11)) - firstDay + 1;
  } else {
    prevMonDate = noOfDays(new Date(date.getFullYear(), date.getMonth() - 1)) - firstDay + 1;
  }

  for (var i = 0; i < noOfWeeks*7; i++) {
    if (i%7 === 0) {
      week = table.appendChild(document.createElement('tr'));
      week.className = 'day';
    }
    
    if (i < firstDay) {
      day = week.appendChild(document.createElement('td'));
      day.innerHTML = prevMonDate++;
      day.className = 'disable';
    } else if (i < sum) {
      day = week.appendChild(document.createElement('td'));
      day.innerHTML = (i - firstDay + 1);

      var newDate = new Date();
      // dosent work as it compares time also
      // if (new Date() === new Date(date.getFullYear(), date.getMonth(), (i - firstDay + 1))) {
      if (
        newDate.getFullYear() === date.getFullYear() &&
        newDate.getMonth() === date.getMonth() &&
        newDate.getDate() === (i - firstDay + 1)
      ) {
        day.className = 'active';
      }
    } else {
      day = week.appendChild(document.createElement('td'));
      day.innerHTML = nextMonthDate++;
      day.className = 'disable';
    }
  }
}

createCalendar();