import {initCode} from './code.mjs';
import {initPlace} from './place.mjs';
import {initNavigation} from './navigation.mjs';
import {initSearch} from './search.mjs';
import {initInterfaces} from './interfaces.mjs';
initCode();await document.fonts.ready;initInterfaces();initPlace();initNavigation();initSearch();

if(!matchMedia("(prefers-reduced-motion: reduce)").matches)document.querySelector(".hero-inner")?.animate([{transform:"translateY(6px)"},{transform:"none"}],{duration:220,easing:"cubic-bezier(.2,.7,.2,1)"});
