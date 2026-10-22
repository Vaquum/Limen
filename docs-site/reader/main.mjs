import {initCode} from './code.mjs';
import {initPlace} from './place.mjs';
import {initNavigation} from './navigation.mjs';
import {initSearch} from './search.mjs';
import {initInterfaces} from './interfaces.mjs';
initCode();await document.fonts.ready;initPlace();initInterfaces();initNavigation();initSearch();
