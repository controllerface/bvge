function handle_event(event)
{
    let target = document.getElementById(event.type);
    if (!target)
    {
        let label = document.createElement('label');
        label.textContent = event.type + " : ";
        target = document.createElement('span');
        target.setAttribute('id', event.type);

        let container = document.createElement('div');
        container.appendChild(label);
        container.appendChild(target);

        let main = document.getElementById('main');
        main.appendChild(container);
    }
    target.textContent = event.data;
}

var event_source = new EventSource("/events");
event_source.addEventListener('dt', handle_event);