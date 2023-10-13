def replace_with_custom_fn_if_matches_filter(
    model, replacement_fn, filter_fn, cur_fqn=''
) -> None:
    """
    For each `child` in `model`, replaces it with `replacement_fn(child)`
    if `filter_fn(child)` is `True`
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        if cur_fqn == '':
            new_fqn = name
        else:
            new_fqn = f'{cur_fqn}.{name}'
        if filter_fn(child, new_fqn):
            new_child = replacement_fn(child)
            setattr(model, name, new_child)
        else:
            replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, new_fqn)
