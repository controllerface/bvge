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
event_source.addEventListener('fps', handle_event);
event_source.addEventListener('phys', handle_event);
event_source.addEventListener('phys_integrate', handle_event);
event_source.addEventListener('phys_bank_offset', handle_event);
event_source.addEventListener('phys_gen_keys', handle_event);
event_source.addEventListener('phys_key_map', handle_event);
event_source.addEventListener('phys_locate_inbounds', handle_event);
event_source.addEventListener('phys_match_candidates', handle_event);
event_source.addEventListener('phys_match_offsets', handle_event);
event_source.addEventListener('phys_aabb_collide', handle_event);
event_source.addEventListener('phys_finalize_candidates', handle_event);
event_source.addEventListener('phys_sat_collide', handle_event);
event_source.addEventListener('phys_sat_scan_reactions', handle_event);
event_source.addEventListener('phys_sat_sort_reactions', handle_event);
event_source.addEventListener('phys_sat_apply_reactions', handle_event);
event_source.addEventListener('phys_move_armatures', handle_event);
event_source.addEventListener('phys_animate_armatures', handle_event);
event_source.addEventListener('phys_animate_bones', handle_event);
event_source.addEventListener('phys_animate_points', handle_event);