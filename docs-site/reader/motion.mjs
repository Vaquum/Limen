const closingDialogs = new WeakMap();

export function openDialog(dialog) {
  closingDialogs.delete(dialog);
  dialog.removeAttribute('data-closing');
  if (!dialog.open) dialog.showModal();
}

export async function closeDialog(dialog, restoreFocus) {
  if (!dialog.open || closingDialogs.has(dialog)) return;

  const closing = Symbol();
  closingDialogs.set(dialog, closing);
  dialog.setAttribute('data-closing', '');
  const transitions = dialog.getAnimations();
  if (transitions.length) await Promise.allSettled(transitions.map(transition => transition.finished));

  if (closingDialogs.get(dialog) !== closing) return;
  closingDialogs.delete(dialog);
  dialog.removeAttribute('data-closing');
  dialog.close();
  restoreFocus?.();
}
