// see http://stackoverflow.com/a/13817235
/*jshint loopfunc:true */
/**
 * Protect window.console method calls, e.g. console is not defined on IE
 * unless dev tools are open, and IE doesn't define console.debug
 */
(function() {
  if (!window.console) {
    window.console = {};
  }
  // union of Chrome, FF, IE, and Safari console methods
  var m = [
    "log", "info", "warn", "error", "debug", "trace", "dir", "group",
    "groupCollapsed", "groupEnd", "time", "timeEnd", "profile", "profileEnd",
    "dirxml", "assert", "count", "markTimeline", "timeStamp", "clear"
  ];
  // define undefined methods as noops to prevent errors
  for (var i = 0; i < m.length; i++) {
    if (!window.console[m[i]]) {
      window.console[m[i]] = function() {};
    }    
  } 
})();/*****************************************************************************
jQuery Placeholder 1.1.1

Copyright (c) 2010 Michael J. Ryan (http://tracker1.info/)

Dual licensed under the MIT and GPL licenses:
	http://www.opensource.org/licenses/mit-license.php
	http://www.gnu.org/licenses/gpl.html

------------------------------------------------------------------------------

Sets up a watermark for inputted fields... this will create a LABEL.watermark 
tag immediately following the input tag, the positioning will be set absolute, 
and it will be positioned to match the input tag.

To activate on all tags with a 'data-watermark' attribute:

	$('input[placeholder],textarea[placeholder]').placeholder();


To style the tags as appropriate (you'll want to make sure the font matches):

	label.placeholder {
		cursor: text;				<--- display a cursor to match the text input

		padding: 4px 4px 4px 4px;   <--- this should match the border+padding 
											for the input send_field(s)
		color: #999999;				<--- this will display as faded
	}

You'll also want to have the color set for browsers with native support
	input:placeholder, textarea:placeholder {
		color: #999999;
	}
	input::-webkit-input-placeholder, textarea::-webkit-input-placeholder {
		color: #999999;
	}

------------------------------------------------------------------------------

Thanks to...
	http://www.alistapart.com/articles/makingcompactformsmoreaccessible
	http://plugins.jquery.com/project/overlabel

	This works similar to the overlabel, but creates the actual label tag
	based on a data-watermark attribute on the input tag, instead of 
	relying on the markup to provide it.

*****************************************************************************/
(function($){
	
	var ph = "PLACEHOLDER-INPUT";
	var phl = "PLACEHOLDER-LABEL";
	var boundEvents = false;
	var default_options = {
		labelClass: 'placeholder'
	};
	
	//check for native support for placeholder attribute, if so stub methods and return
	var input = document.createElement("input");
	if ('placeholder' in input) {
		$.fn.placeholder = $.fn.unplaceholder = function(){}; //empty function
		delete input; //cleanup IE memory
		return;
	};
	delete input;

	$.fn.placeholder = function(options) {
		bindEvents();

		var opts = $.extend(default_options, options)

		this.each(function(){
			var rnd=Math.random().toString(32).replace(/\./,'')
				,input=$(this)
				,label=$('<label style="position:absolute;display:none;top:0;left:0;"></label>');

			if (!input.attr('placeholder') || input.data(ph) === ph) return; //already watermarked

			//make sure the input tag has an ID assigned, if not, assign one.
			if (!input.attr('id')) input.attr('id') = 'input_' + rnd;

			label	.attr('id',input.attr('id') + "_placeholder")
					.data(ph, '#' + input.attr('id'))	//reference to the input tag
					.attr('for',input.attr('id'))
					.addClass(opts.labelClass)
					.addClass(opts.labelClass + '-for-' + this.tagName.toLowerCase()) //ex: watermark-for-textarea
					.addClass(phl)
					.text(input.attr('placeholder'));

			input
				.data(phl, '#' + label.attr('id'))	//set a reference to the label
				.data(ph,ph)		//set that the send_field is watermarked
				.addClass(ph)		//add the watermark class
				.after(label);		//add the label send_field to the page

			//setup overlay
			itemIn.call(this);
			itemOut.call(this);
		});
	};

	$.fn.unplaceholder = function(){
		this.each(function(){
			var	input=$(this),
				label=$(input.data(phl));

			if (input.data(ph) !== ph) return;
				
			label.remove();
			input.removeData(ph).removeData(phl).removeClass(ph);
		});
	};


	function bindEvents() {
		if (boundEvents) return;

		//prepare live bindings if not already done.
		$('.' + ph)
			.live('click',itemIn)
			.live('focusin',itemIn)
			.live('focusout',itemOut);
		bound = true;

		boundEvents = true;
	};

	function itemIn() {
		var input = $(this)
			,label = $(input.data(phl));

		label.css('display', 'none');
	};

	function itemOut() {
		var that = this;

		//use timeout to let other validators/formatters directly bound to blur/focusout work first
		setTimeout(function(){
			var input = $(that);
			$(input.data(phl))
				.css('top', input.position().top + 23 + 'px')
				.css('left', input.position().left + 7 + 'px')
				.css('display', !!input.val() ? 'none' : 'block');
		}, 200);
	};

}(jQuery));/*!
 * jQuery blockUI plugin
 * Version 2.57.0-2013.02.17
 * @requires jQuery v1.7 or later
 *
 * Examples at: http://malsup.com/jquery/block/
 * Copyright (c) 2007-2013 M. Alsup
 * Dual licensed under the MIT and GPL licenses:
 * http://www.opensource.org/licenses/mit-license.php
 * http://www.gnu.org/licenses/gpl.html
 *
 * Thanks to Amir-Hossein Sobhi for some excellent contributions!
 */

;(function() {
/*jshint eqeqeq:false curly:false latedef:false */
"use strict";

	function setup($) {
		$.fn._fadeIn = $.fn.fadeIn;

		var noOp = $.noop || function() {};

		// this bit is to ensure we don't call setExpression when we shouldn't (with extra muscle to handle
		// retarded userAgent strings on Vista)
		var msie = /MSIE/.test(navigator.userAgent);
		var ie6  = /MSIE 6.0/.test(navigator.userAgent) && ! /MSIE 8.0/.test(navigator.userAgent);
		var mode = document.documentMode || 0;
		// var setExpr = msie && (($.browser.version < 8 && !mode) || mode < 8);
		var setExpr = $.isFunction( document.createElement('div').style.setExpression );

		// global $ methods for blocking/unblocking the entire page
		$.blockUI   = function(opts) { install(window, opts); };
		$.unblockUI = function(opts) { remove(window, opts); };

		// convenience method for quick growl-like notifications  (http://www.google.com/search?q=growl)
		$.growlUI = function(title, message, timeout, onClose) {
			var $m = $('<div class="growlUI"></div>');
			if (title) $m.append('<h1>'+title+'</h1>');
			if (message) $m.append('<h2>'+message+'</h2>');
			if (timeout === undefined) timeout = 3000;
			$.blockUI({
				message: $m, fadeIn: 700, fadeOut: 1000, centerY: false,
				timeout: timeout, showOverlay: false,
				onUnblock: onClose,
				css: $.blockUI.defaults.growlCSS
			});
		};

		// plugin method for blocking element content
		$.fn.block = function(opts) {
			var fullOpts = $.extend({}, $.blockUI.defaults, opts || {});
			this.each(function() {
				var $el = $(this);
				if (fullOpts.ignoreIfBlocked && $el.data('blockUI.isBlocked'))
					return;
				$el.unblock({ fadeOut: 0 });
			});

			return this.each(function() {
				if ($.css(this,'position') == 'static') {
					this.style.position = 'relative';
					$(this).data('blockUI.static', true);
				}
				this.style.zoom = 1; // force 'hasLayout' in ie
				install(this, opts);
			});
		};

		// plugin method for unblocking element content
		$.fn.unblock = function(opts) {
			return this.each(function() {
				remove(this, opts);
			});
		};

		$.blockUI.version = 2.57; // 2nd generation blocking at no extra cost!

		// override these in your code to change the default behavior and style
		$.blockUI.defaults = {
			// message displayed when blocking (use null for no message)
			message:  '<h1>Please wait...</h1>',

			title: null,		// title string; only used when theme == true
			draggable: true,	// only used when theme == true (requires jquery-ui.js to be loaded)

			theme: false, // set to true to use with jQuery UI themes

			// styles for the message when blocking; if you wish to disable
			// these and use an external stylesheet then do this in your code:
			// $.blockUI.defaults.css = {};
			css: {
				padding:	0,
				margin:		0,
				width:		'30%',
				top:		'40%',
				left:		'35%',
				textAlign:	'center',
				color:		'#000',
				border:		'3px solid #aaa',
				backgroundColor:'#fff',
				cursor:		'wait'
			},

			// minimal style set used when themes are used
			themedCSS: {
				width:	'30%',
				top:	'40%',
				left:	'35%'
			},

			// styles for the overlay
			overlayCSS:  {
				backgroundColor:	'#000',
				opacity:			0.6,
				cursor:				'wait'
			},

			// style to replace wait cursor before unblocking to correct issue
			// of lingering wait cursor
			cursorReset: 'default',

			// styles applied when using $.growlUI
			growlCSS: {
				width:		'350px',
				top:		'10px',
				left:		'',
				right:		'10px',
				border:		'none',
				padding:	'5px',
				opacity:	0.6,
				cursor:		'default',
				color:		'#fff',
				backgroundColor: '#000',
				'-webkit-border-radius':'10px',
				'-moz-border-radius':	'10px',
				'border-radius':		'10px'
			},

			// IE issues: 'about:blank' fails on HTTPS and javascript:false is s-l-o-w
			// (hat tip to Jorge H. N. de Vasconcelos)
			/*jshint scripturl:true */
			iframeSrc: /^https/i.test(window.location.href || '') ? 'javascript:false' : 'about:blank',

			// force usage of iframe in non-IE browsers (handy for blocking applets)
			forceIframe: false,

			// z-index for the blocking overlay
			baseZ: 1000,

			// set these to true to have the message automatically centered
			centerX: true, // <-- only effects element blocking (page block controlled via css above)
			centerY: true,

			// allow body element to be stetched in ie6; this makes blocking look better
			// on "short" pages.  disable if you wish to prevent changes to the body height
			allowBodyStretch: true,

			// enable if you want key and mouse events to be disabled for content that is blocked
			bindEvents: true,

			// be default blockUI will supress tab navigation from leaving blocking content
			// (if bindEvents is true)
			constrainTabKey: true,

			// fadeIn time in millis; set to 0 to disable fadeIn on block
			fadeIn:  200,

			// fadeOut time in millis; set to 0 to disable fadeOut on unblock
			fadeOut:  400,

			// time in millis to wait before auto-unblocking; set to 0 to disable auto-unblock
			timeout: 0,

			// disable if you don't want to show the overlay
			showOverlay: true,

			// if true, focus will be placed in the first available input send_field when
			// page blocking
			focusInput: true,

			// suppresses the use of overlay styles on FF/Linux (due to performance issues with opacity)
			// no longer needed in 2012
			// applyPlatformOpacityRules: true,

			// callback method invoked when fadeIn has completed and blocking message is visible
			onBlock: null,

			// callback method invoked when unblocking has completed; the callback is
			// passed the element that has been unblocked (which is the window object for page
			// blocks) and the options that were passed to the unblock call:
			//	onUnblock(element, options)
			onUnblock: null,

			// callback method invoked when the overlay area is clicked.
			// setting this will turn the cursor to a pointer, otherwise cursor defined in overlayCss will be used.
			onOverlayClick: null,

			// don't ask; if you really must know: http://groups.google.com/group/jquery-en/browse_thread/thread/36640a8730503595/2f6a79a77a78e493#2f6a79a77a78e493
			quirksmodeOffsetHack: 4,

			// class name of the message block
			blockMsgClass: 'blockMsg',

			// if it is already blocked, then ignore it (don't unblock and reblock)
			ignoreIfBlocked: false
		};

		// private data and functions follow...

		var pageBlock = null;
		var pageBlockEls = [];

		function install(el, opts) {
			var css, themedCSS;
			var full = (el == window);
			var msg = (opts && opts.message !== undefined ? opts.message : undefined);
			opts = $.extend({}, $.blockUI.defaults, opts || {});

			if (opts.ignoreIfBlocked && $(el).data('blockUI.isBlocked'))
				return;

			opts.overlayCSS = $.extend({}, $.blockUI.defaults.overlayCSS, opts.overlayCSS || {});
			css = $.extend({}, $.blockUI.defaults.css, opts.css || {});
			if (opts.onOverlayClick)
				opts.overlayCSS.cursor = 'pointer';

			themedCSS = $.extend({}, $.blockUI.defaults.themedCSS, opts.themedCSS || {});
			msg = msg === undefined ? opts.message : msg;

			// remove the current block (if there is one)
			if (full && pageBlock)
				remove(window, {fadeOut:0});

			// if an existing element is being used as the blocking content then we capture
			// its current place in the DOM (and current display style) so we can restore
			// it when we unblock
			if (msg && typeof msg != 'string' && (msg.parentNode || msg.jquery)) {
				var node = msg.jquery ? msg[0] : msg;
				var data = {};
				$(el).data('blockUI.history', data);
				data.el = node;
				data.parent = node.parentNode;
				data.display = node.style.display;
				data.position = node.style.position;
				if (data.parent)
					data.parent.removeChild(node);
			}

			$(el).data('blockUI.onUnblock', opts.onUnblock);
			var z = opts.baseZ;

			// blockUI uses 3 layers for blocking, for simplicity they are all used on every platform;
			// layer1 is the iframe layer which is used to supress bleed through of underlying content
			// layer2 is the overlay layer which has opacity and a wait cursor (by default)
			// layer3 is the message content that is displayed while blocking
			var lyr1, lyr2, lyr3, s;
			if (msie || opts.forceIframe)
				lyr1 = $('<iframe class="blockUI" style="z-index:'+ (z++) +';display:none;border:none;margin:0;padding:0;position:absolute;width:100%;height:100%;top:0;left:0" src="'+opts.iframeSrc+'"></iframe>');
			else
				lyr1 = $('<div class="blockUI" style="display:none"></div>');

			if (opts.theme)
				lyr2 = $('<div class="blockUI blockOverlay ui-widget-overlay" style="z-index:'+ (z++) +';display:none"></div>');
			else
				lyr2 = $('<div class="blockUI blockOverlay" style="z-index:'+ (z++) +';display:none;border:none;margin:0;padding:0;width:100%;height:100%;top:0;left:0"></div>');

			if (opts.theme && full) {
				s = '<div class="blockUI ' + opts.blockMsgClass + ' blockPage ui-dialog ui-widget ui-corner-all" style="z-index:'+(z+10)+';display:none;position:fixed">';
				if ( opts.title ) {
					s += '<div class="ui-widget-header ui-dialog-titlebar ui-corner-all blockTitle">'+(opts.title || '&nbsp;')+'</div>';
				}
				s += '<div class="ui-widget-content ui-dialog-content"></div>';
				s += '</div>';
			}
			else if (opts.theme) {
				s = '<div class="blockUI ' + opts.blockMsgClass + ' blockElement ui-dialog ui-widget ui-corner-all" style="z-index:'+(z+10)+';display:none;position:absolute">';
				if ( opts.title ) {
					s += '<div class="ui-widget-header ui-dialog-titlebar ui-corner-all blockTitle">'+(opts.title || '&nbsp;')+'</div>';
				}  
				s += '<div class="ui-widget-content ui-dialog-content"></div>';
				s += '</div>';
			}
			else if (full) {
				s = '<div class="blockUI ' + opts.blockMsgClass + ' blockPage" style="z-index:'+(z+10)+';display:none;position:fixed"></div>';
			}
			else {
				s = '<div class="blockUI ' + opts.blockMsgClass + ' blockElement" style="z-index:'+(z+10)+';display:none;position:absolute"></div>';
			}
			lyr3 = $(s);

			// if we have a message, style it
			if (msg) {
				if (opts.theme) {
					lyr3.css(themedCSS);
					lyr3.addClass('ui-widget-content');
				}
				else
					lyr3.css(css);
			}

			// style the overlay
			if (!opts.theme /*&& (!opts.applyPlatformOpacityRules)*/)
				lyr2.css(opts.overlayCSS);
			lyr2.css('position', full ? 'fixed' : 'absolute');

			// make iframe layer transparent in IE
			if (msie || opts.forceIframe)
				lyr1.css('opacity',0.0);

			//$([lyr1[0],lyr2[0],lyr3[0]]).appendTo(full ? 'body' : el);
			var layers = [lyr1,lyr2,lyr3], $par = full ? $('body') : $(el);
			$.each(layers, function() {
				this.appendTo($par);
			});

			if (opts.theme && opts.draggable && $.fn.draggable) {
				lyr3.draggable({
					handle: '.ui-dialog-titlebar',
					cancel: 'li'
				});
			}

			// ie7 must use absolute positioning in quirks mode and to account for activex issues (when scrolling)
			var expr = setExpr && (!$.support.boxModel || $('object,embed', full ? null : el).length > 0);
			if (ie6 || expr) {
				// give body 100% height
				if (full && opts.allowBodyStretch && $.support.boxModel)
					$('html,body').css('height','100%');

				// fix ie6 issue when blocked element has a border width
				if ((ie6 || !$.support.boxModel) && !full) {
					var t = sz(el,'borderTopWidth'), l = sz(el,'borderLeftWidth');
					var fixT = t ? '(0 - '+t+')' : 0;
					var fixL = l ? '(0 - '+l+')' : 0;
				}

				// simulate fixed position
				$.each(layers, function(i,o) {
					var s = o[0].style;
					s.position = 'absolute';
					if (i < 2) {
						if (full)
							s.setExpression('height','Math.max(document.body.scrollHeight, document.body.offsetHeight) - (jQuery.support.boxModel?0:'+opts.quirksmodeOffsetHack+') + "px"');
						else
							s.setExpression('height','this.parentNode.offsetHeight + "px"');
						if (full)
							s.setExpression('width','jQuery.support.boxModel && document.documentElement.clientWidth || document.body.clientWidth + "px"');
						else
							s.setExpression('width','this.parentNode.offsetWidth + "px"');
						if (fixL) s.setExpression('left', fixL);
						if (fixT) s.setExpression('top', fixT);
					}
					else if (opts.centerY) {
						if (full) s.setExpression('top','(document.documentElement.clientHeight || document.body.clientHeight) / 2 - (this.offsetHeight / 2) + (blah = document.documentElement.scrollTop ? document.documentElement.scrollTop : document.body.scrollTop) + "px"');
						s.marginTop = 0;
					}
					else if (!opts.centerY && full) {
						var top = (opts.css && opts.css.top) ? parseInt(opts.css.top, 10) : 0;
						var expression = '((document.documentElement.scrollTop ? document.documentElement.scrollTop : document.body.scrollTop) + '+top+') + "px"';
						s.setExpression('top',expression);
					}
				});
			}

			// show the message
			if (msg) {
				if (opts.theme)
					lyr3.find('.ui-widget-content').append(msg);
				else
					lyr3.append(msg);
				if (msg.jquery || msg.nodeType)
					$(msg).show();
			}

			if ((msie || opts.forceIframe) && opts.showOverlay)
				lyr1.show(); // opacity is zero
			if (opts.fadeIn) {
				var cb = opts.onBlock ? opts.onBlock : noOp;
				var cb1 = (opts.showOverlay && !msg) ? cb : noOp;
				var cb2 = msg ? cb : noOp;
				if (opts.showOverlay)
					lyr2._fadeIn(opts.fadeIn, cb1);
				if (msg)
					lyr3._fadeIn(opts.fadeIn, cb2);
			}
			else {
				if (opts.showOverlay)
					lyr2.show();
				if (msg)
					lyr3.show();
				if (opts.onBlock)
					opts.onBlock();
			}

			// bind key and mouse events
			bind(1, el, opts);

			if (full) {
				pageBlock = lyr3[0];
				pageBlockEls = $(':input:enabled:visible',pageBlock);
				if (opts.focusInput)
					setTimeout(focus, 20);
			}
			else
				center(lyr3[0], opts.centerX, opts.centerY);

			if (opts.timeout) {
				// auto-unblock
				var to = setTimeout(function() {
					if (full)
						$.unblockUI(opts);
					else
						$(el).unblock(opts);
				}, opts.timeout);
				$(el).data('blockUI.timeout', to);
			}
		}

		// remove the block
		function remove(el, opts) {
			var full = (el == window);
			var $el = $(el);
			var data = $el.data('blockUI.history');
			var to = $el.data('blockUI.timeout');
			if (to) {
				clearTimeout(to);
				$el.removeData('blockUI.timeout');
			}
			opts = $.extend({}, $.blockUI.defaults, opts || {});
			bind(0, el, opts); // unbind events

			if (opts.onUnblock === null) {
				opts.onUnblock = $el.data('blockUI.onUnblock');
				$el.removeData('blockUI.onUnblock');
			}

			var els;
			if (full) // crazy selector to handle odd send_field errors in ie6/7
				els = $('body').children().filter('.blockUI').add('body > .blockUI');
			else
				els = $el.find('>.blockUI');

			// fix cursor issue
			if ( opts.cursorReset ) {
				if ( els.length > 1 )
					els[1].style.cursor = opts.cursorReset;
				if ( els.length > 2 )
					els[2].style.cursor = opts.cursorReset;
			}

			if (full)
				pageBlock = pageBlockEls = null;

			if (opts.fadeOut) {
				els.fadeOut(opts.fadeOut);
				setTimeout(function() { reset(els,data,opts,el); }, opts.fadeOut);
			}
			else
				reset(els, data, opts, el);
		}

		// move blocking element back into the DOM where it started
		function reset(els,data,opts,el) {
			var $el = $(el);
			els.each(function(i,o) {
				// remove via DOM calls so we don't lose event handlers
				if (this.parentNode)
					this.parentNode.removeChild(this);
			});

			if (data && data.el) {
				data.el.style.display = data.display;
				data.el.style.position = data.position;
				if (data.parent)
					data.parent.appendChild(data.el);
				$el.removeData('blockUI.history');
			}

			if ($el.data('blockUI.static')) {
				$el.css('position', 'static'); // #22
			}

			if (typeof opts.onUnblock == 'function')
				opts.onUnblock(el,opts);

			// fix issue in Safari 6 where block artifacts remain until reflow
			var body = $(document.body), w = body.width(), cssW = body[0].style.width;
			body.width(w-1).width(w);
			body[0].style.width = cssW;
		}

		// bind/unbind the handler
		function bind(b, el, opts) {
			var full = el == window, $el = $(el);

			// don't bother unbinding if there is nothing to unbind
			if (!b && (full && !pageBlock || !full && !$el.data('blockUI.isBlocked')))
				return;

			$el.data('blockUI.isBlocked', b);

			// don't bind events when overlay is not in use or if bindEvents is false
			if (!opts.bindEvents || (b && !opts.showOverlay))
				return;

			// bind anchors and inputs for mouse and key events
			var events = 'mousedown mouseup keydown keypress keyup touchstart touchend touchmove';
			if (b)
				$(document).bind(events, opts, handler);
			else
				$(document).unbind(events, handler);

		// former impl...
		//		var $e = $('a,:input');
		//		b ? $e.bind(events, opts, handler) : $e.unbind(events, handler);
		}

		// event handler to suppress keyboard/mouse events when blocking
		function handler(e) {
			// allow tab navigation (conditionally)
			if (e.keyCode && e.keyCode == 9) {
				if (pageBlock && e.data.constrainTabKey) {
					var els = pageBlockEls;
					var fwd = !e.shiftKey && e.target === els[els.length-1];
					var back = e.shiftKey && e.target === els[0];
					if (fwd || back) {
						setTimeout(function(){focus(back);},10);
						return false;
					}
				}
			}
			var opts = e.data;
			var target = $(e.target);
			if (target.hasClass('blockOverlay') && opts.onOverlayClick)
				opts.onOverlayClick();

			// allow events within the message content
			if (target.parents('div.' + opts.blockMsgClass).length > 0)
				return true;

			// allow events for content that is not being blocked
			return target.parents().children().filter('div.blockUI').length === 0;
		}

		function focus(back) {
			if (!pageBlockEls)
				return;
			var e = pageBlockEls[back===true ? pageBlockEls.length-1 : 0];
			if (e)
				e.focus();
		}

		function center(el, x, y) {
			var p = el.parentNode, s = el.style;
			var l = ((p.offsetWidth - el.offsetWidth)/2) - sz(p,'borderLeftWidth');
			var t = ((p.offsetHeight - el.offsetHeight)/2) - sz(p,'borderTopWidth');
			if (x) s.left = l > 0 ? (l+'px') : '0';
			if (y) s.top  = t > 0 ? (t+'px') : '0';
		}

		function sz(el, p) {
			return parseInt($.css(el,p),10)||0;
		}

	}


	/*global define:true */
	if (typeof define === 'function' && define.amd && define.amd.jQuery) {
		define(['jquery'], setup);
	} else {
		setup(jQuery);
	}

})();
// Chosen, a Select Box Enhancer for jQuery and Protoype
// by Patrick Filler for Harvest, http://getharvest.com
//
// Version 0.9.11
// Full source at https://github.com/harvesthq/chosen
// Copyright (c) 2011 Harvest http://getharvest.com

// MIT License, https://github.com/harvesthq/chosen/blob/master/LICENSE.md
// This file is generated by `cake build`, do not edit it by hand.
(function() {
  var SelectParser;

  SelectParser = (function() {

    function SelectParser() {
      this.options_index = 0;
      this.parsed = [];
    }

    SelectParser.prototype.add_node = function(child) {
      if (child.nodeName.toUpperCase() === "OPTGROUP") {
        return this.add_group(child);
      } else {
        return this.add_option(child);
      }
    };

    SelectParser.prototype.add_group = function(group) {
      var group_position, option, _i, _len, _ref, _results;
      group_position = this.parsed.length;
      this.parsed.push({
        array_index: group_position,
        group: true,
        label: group.label,
        children: 0,
        disabled: group.disabled
      });
      _ref = group.childNodes;
      _results = [];
      for (_i = 0, _len = _ref.length; _i < _len; _i++) {
        option = _ref[_i];
        _results.push(this.add_option(option, group_position, group.disabled));
      }
      return _results;
    };

    SelectParser.prototype.add_option = function(option, group_position, group_disabled) {
      if (option.nodeName.toUpperCase() === "OPTION") {
        if (option.text !== "") {
          if (group_position != null) {
            this.parsed[group_position].children += 1;
          }
          this.parsed.push({
            array_index: this.parsed.length,
            options_index: this.options_index,
            value: option.value,
            text: option.text,
            html: option.innerHTML,
            selected: option.selected,
            disabled: group_disabled === true ? group_disabled : option.disabled,
            group_array_index: group_position,
            classes: option.className,
            style: option.style.cssText
          });
        } else {
          this.parsed.push({
            array_index: this.parsed.length,
            options_index: this.options_index,
            empty: true
          });
        }
        return this.options_index += 1;
      }
    };

    return SelectParser;

  })();

  SelectParser.select_to_array = function(select) {
    var child, parser, _i, _len, _ref;
    parser = new SelectParser();
    _ref = select.childNodes;
    for (_i = 0, _len = _ref.length; _i < _len; _i++) {
      child = _ref[_i];
      parser.add_node(child);
    }
    return parser.parsed;
  };

  this.SelectParser = SelectParser;

}).call(this);

/*
Chosen source: generate output using 'cake build'
Copyright (c) 2011 by Harvest
*/


(function() {
  var AbstractChosen, root;

  root = this;

  AbstractChosen = (function() {

    function AbstractChosen(form_field, options) {
      this.form_field = form_field;
      this.options = options != null ? options : {};
      this.is_multiple = this.form_field.multiple;
      this.set_default_text();
      this.set_default_values();
      this.setup();
      this.set_up_html();
      this.register_observers();
      this.finish_setup();
    }

    AbstractChosen.prototype.set_default_values = function() {
      var _this = this;
      this.click_test_action = function(evt) {
        return _this.test_active_click(evt);
      };
      this.activate_action = function(evt) {
        return _this.activate_field(evt);
      };
      this.active_field = false;
      this.mouse_on_container = false;
      this.results_showing = false;
      this.result_highlighted = null;
      this.result_single_selected = null;
      this.allow_single_deselect = (this.options.allow_single_deselect != null) && (this.form_field.options[0] != null) && this.form_field.options[0].text === "" ? this.options.allow_single_deselect : false;
      this.disable_search_threshold = this.options.disable_search_threshold || 0;
      this.disable_search = this.options.disable_search || false;
      this.enable_split_word_search = this.options.enable_split_word_search != null ? this.options.enable_split_word_search : true;
      this.search_contains = this.options.search_contains || false;
      this.choices = 0;
      this.single_backstroke_delete = this.options.single_backstroke_delete || false;
      this.max_selected_options = this.options.max_selected_options || Infinity;
      return this.inherit_select_classes = this.options.inherit_select_classes || false;
    };

    AbstractChosen.prototype.set_default_text = function() {
      if (this.form_field.getAttribute("data-placeholder")) {
        this.default_text = this.form_field.getAttribute("data-placeholder");
      } else if (this.is_multiple) {
        this.default_text = this.options.placeholder_text_multiple || this.options.placeholder_text || "Select Some Options";
      } else {
        this.default_text = this.options.placeholder_text_single || this.options.placeholder_text || "Select an Option";
      }
      return this.results_none_found = this.form_field.getAttribute("data-no_results_text") || this.options.no_results_text || "No results match";
    };

    AbstractChosen.prototype.mouse_enter = function() {
      return this.mouse_on_container = true;
    };

    AbstractChosen.prototype.mouse_leave = function() {
      return this.mouse_on_container = false;
    };

    AbstractChosen.prototype.input_focus = function(evt) {
      var _this = this;
      if (this.is_multiple) {
        if (!this.active_field) {
          return setTimeout((function() {
            return _this.container_mousedown();
          }), 50);
        }
      } else {
        if (!this.active_field) {
          return this.activate_field();
        }
      }
    };

    AbstractChosen.prototype.input_blur = function(evt) {
      var _this = this;
      if (!this.mouse_on_container) {
        this.active_field = false;
        return setTimeout((function() {
          return _this.blur_test();
        }), 100);
      }
    };

    AbstractChosen.prototype.result_add_option = function(option) {
      var classes, style;
      if (!option.disabled) {
        option.dom_id = this.container_id + "_o_" + option.array_index;
        classes = option.selected && this.is_multiple ? [] : ["active-result"];
        if (option.selected) {
          classes.push("result-selected");
        }
        if (option.group_array_index != null) {
          classes.push("group-option");
        }
        if (option.classes !== "") {
          classes.push(option.classes);
        }
        style = option.style.cssText !== "" ? " style=\"" + option.style + "\"" : "";
        return '<li id="' + option.dom_id + '" class="' + classes.join(' ') + '"' + style + '>' + option.html + '</li>';
      } else {
        return "";
      }
    };

    AbstractChosen.prototype.results_update_field = function() {
      if (!this.is_multiple) {
        this.results_reset_cleanup();
      }
      this.result_clear_highlight();
      this.result_single_selected = null;
      return this.results_build();
    };

    AbstractChosen.prototype.results_toggle = function() {
      if (this.results_showing) {
        return this.results_hide();
      } else {
        return this.results_show();
      }
    };

    AbstractChosen.prototype.results_search = function(evt) {
      if (this.results_showing) {
        return this.winnow_results();
      } else {
        return this.results_show();
      }
    };

    AbstractChosen.prototype.keyup_checker = function(evt) {
      var stroke, _ref;
      stroke = (_ref = evt.which) != null ? _ref : evt.keyCode;
      this.search_field_scale();
      switch (stroke) {
        case 8:
          if (this.is_multiple && this.backstroke_length < 1 && this.choices > 0) {
            return this.keydown_backstroke();
          } else if (!this.pending_backstroke) {
            this.result_clear_highlight();
            return this.results_search();
          }
          break;
        case 13:
          evt.preventDefault();
          if (this.results_showing) {
            return this.result_select(evt);
          }
          break;
        case 27:
          if (this.results_showing) {
            this.results_hide();
          }
          return true;
        case 9:
        case 38:
        case 40:
        case 16:
        case 91:
        case 17:
          break;
        default:
          return this.results_search();
      }
    };

    AbstractChosen.prototype.generate_field_id = function() {
      var new_id;
      new_id = this.generate_random_id();
      this.form_field.id = new_id;
      return new_id;
    };

    AbstractChosen.prototype.generate_random_char = function() {
      var chars, newchar, rand;
      chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
      rand = Math.floor(Math.random() * chars.length);
      return newchar = chars.substring(rand, rand + 1);
    };

    AbstractChosen.prototype.fixup_width = function(width) {
      var format_regex;
      format_regex = new RegExp('(px|em|ex|%|in|cm|mm|pt|pc)$', 'i');
      if (!format_regex.test(width)) {
        return "" + width + "px";
      }
      return width;
    };

    return AbstractChosen;

  })();

  root.AbstractChosen = AbstractChosen;

}).call(this);

/*
Chosen source: generate output using 'cake build'
Copyright (c) 2011 by Harvest
*/


(function() {
  var $, Chosen, get_side_border_padding, root,
    __hasProp = {}.hasOwnProperty,
    __extends = function(child, parent) { for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; };

  root = this;

  $ = jQuery;

  $.fn.extend({
    chosen: function(options) {
      var browser, match, ua;
      ua = navigator.userAgent.toLowerCase();
      match = /(msie) ([\w.]+)/.exec(ua) || [];
      browser = {
        name: match[1] || "",
        version: match[2] || "0"
      };
      if (browser.name === "msie" && (browser.version === "6.0" || (browser.version === "7.0" && document.documentMode === 7))) {
        return this;
      }
      return this.each(function(input_field) {
        var $this;
        $this = $(this);
        if (!$this.hasClass("chzn-done")) {
          return $this.data('chosen', new Chosen(this, options));
        }
      });
    }
  });

  Chosen = (function(_super) {

    __extends(Chosen, _super);

    function Chosen() {
      return Chosen.__super__.constructor.apply(this, arguments);
    }

    Chosen.prototype.setup = function() {
      this.form_field_jq = $(this.form_field);
      this.current_value = this.form_field_jq.val();
      return this.is_rtl = this.form_field_jq.hasClass("chzn-rtl");
    };

    Chosen.prototype.finish_setup = function() {
      return this.form_field_jq.addClass("chzn-done");
    };

    Chosen.prototype.set_up_html = function() {
      var container_classes, container_div, container_props, dd_top, dd_width, sf_width;
      this.container_id = this.form_field.id.length ? this.form_field.id.replace(/[^\w]/g, '_') : this.generate_field_id();
      this.container_id += "_chzn";
      container_classes = ["chzn-container"];
      container_classes.push("chzn-container-" + (this.is_multiple ? "multi" : "single"));
      if (this.inherit_select_classes && this.form_field.className) {
        container_classes.push(this.form_field.className);
      }
      if (this.is_rtl) {
        container_classes.push("chzn-rtl");
      }
			// this.f_width = this.form_field_jq.outerWidth();
      this.f_width = this.fixup_width(this.options.width ? this.options.width : this.f_width);
      container_props = {
        id: this.container_id,
        "class": container_classes.join(' '),
				// style: 'width: ' + this.f_width + 'px;',
        style: 'width: ' + this.f_width,
        title: this.form_field.title
      };
      container_div = $("<div />", container_props);
      if (this.is_multiple) {
        container_div.html('<ul class="chzn-choices"><li class="search-send_field"><input type="text" value="' + this.default_text + '" class="default" autocomplete="off" style="width:25px;" /></li></ul><div class="chzn-drop" style="left:-9000px;"><ul class="chzn-results"></ul></div>');
      } else {
        container_div.html('<a href="javascript:void(0)" class="chzn-single chzn-default" tabindex="-1"><span>' + this.default_text + '</span><div><b></b></div></a><div class="chzn-drop" style="left:-9000px;"><div class="chzn-search"><input type="text" autocomplete="off" /></div><ul class="chzn-results"></ul></div>');
      }
      this.form_field_jq.hide().after(container_div);
      this.container = $('#' + this.container_id);
      this.dropdown = this.container.find('div.chzn-drop').first();
      dd_top = this.container.height();
      dd_width = this.fixup_width(this.options.width ? this.options.width : this.f_width - get_side_border_padding(this.dropdown));
      
      
      var percent_regex = new RegExp('(%)$', 'i');
      if (percent_regex.test(dd_width)) {
      	dd_width = parseFloat(100) -0.3 + "%"
      }
      
      this.dropdown.css({
        "width": dd_width,
        "top": dd_top + "px"
      });
      this.search_field = this.container.find('input').first();
      this.search_results = this.container.find('ul.chzn-results').first();
      this.search_field_scale();
      this.search_no_results = this.container.find('li.no-results').first();
      if (this.is_multiple) {
        this.search_choices = this.container.find('ul.chzn-choices').first();
        this.search_container = this.container.find('li.search-send_field').first();
      } else {
        this.search_container = this.container.find('div.chzn-search').first();
        this.selected_item = this.container.find('.chzn-single').first();
        sf_width = dd_width - get_side_border_padding(this.search_container) - get_side_border_padding(this.search_field);
        this.search_field.css({
          "width": sf_width + "px"
        });
      }
      this.results_build();
      this.set_tab_index();
      return this.form_field_jq.trigger("liszt:ready", {
        chosen: this
      });
    };

    Chosen.prototype.register_observers = function() {
      var _this = this;
      this.container.mousedown(function(evt) {
        return _this.container_mousedown(evt);
      });
      this.container.mouseup(function(evt) {
        return _this.container_mouseup(evt);
      });
      this.container.mouseenter(function(evt) {
        return _this.mouse_enter(evt);
      });
      this.container.mouseleave(function(evt) {
        return _this.mouse_leave(evt);
      });
      this.search_results.mouseup(function(evt) {
        return _this.search_results_mouseup(evt);
      });
      this.search_results.mouseover(function(evt) {
        return _this.search_results_mouseover(evt);
      });
      this.search_results.mouseout(function(evt) {
        return _this.search_results_mouseout(evt);
      });
      this.form_field_jq.bind("liszt:updated", function(evt) {
        return _this.results_update_field(evt);
      });
      this.form_field_jq.bind("liszt:activate", function(evt) {
        return _this.activate_field(evt);
      });
      this.form_field_jq.bind("liszt:open", function(evt) {
        return _this.container_mousedown(evt);
      });
      this.search_field.blur(function(evt) {
        return _this.input_blur(evt);
      });
      this.search_field.keyup(function(evt) {
        return _this.keyup_checker(evt);
      });
      this.search_field.keydown(function(evt) {
        return _this.keydown_checker(evt);
      });
      this.search_field.focus(function(evt) {
        return _this.input_focus(evt);
      });
      if (this.is_multiple) {
        return this.search_choices.click(function(evt) {
          return _this.choices_click(evt);
        });
      } else {
        return this.container.click(function(evt) {
          return evt.preventDefault();
        });
      }
    };

    Chosen.prototype.search_field_disabled = function() {
      this.is_disabled = this.form_field_jq[0].disabled;
      if (this.is_disabled) {
        this.container.addClass('chzn-disabled');
        this.search_field[0].disabled = true;
        if (!this.is_multiple) {
          this.selected_item.unbind("focus", this.activate_action);
        }
        return this.close_field();
      } else {
        this.container.removeClass('chzn-disabled');
        this.search_field[0].disabled = false;
        if (!this.is_multiple) {
          return this.selected_item.bind("focus", this.activate_action);
        }
      }
    };

    Chosen.prototype.container_mousedown = function(evt) {
      var target_closelink;
      if (!this.is_disabled) {
        target_closelink = evt != null ? ($(evt.target)).hasClass("search-choice-close") : false;
        if (evt && evt.type === "mousedown" && !this.results_showing) {
          evt.preventDefault();
        }
        if (!this.pending_destroy_click && !target_closelink) {
          if (!this.active_field) {
            if (this.is_multiple) {
              this.search_field.val("");
            }
            $(document).click(this.click_test_action);
            this.results_show();
          } else if (!this.is_multiple && evt && (($(evt.target)[0] === this.selected_item[0]) || $(evt.target).parents("a.chzn-single").length)) {
            evt.preventDefault();
            this.results_toggle();
          }
          return this.activate_field();
        } else {
          return this.pending_destroy_click = false;
        }
      }
    };

    Chosen.prototype.container_mouseup = function(evt) {
      if (evt.target.nodeName === "ABBR" && !this.is_disabled) {
        return this.results_reset(evt);
      }
    };

    Chosen.prototype.blur_test = function(evt) {
      if (!this.active_field && this.container.hasClass("chzn-container-active")) {
        return this.close_field();
      }
    };

    Chosen.prototype.close_field = function() {
      $(document).unbind("click", this.click_test_action);
      this.active_field = false;
      this.results_hide();
      this.container.removeClass("chzn-container-active");
      this.winnow_results_clear();
      this.clear_backstroke();
      this.show_search_field_default();
      return this.search_field_scale();
    };

    Chosen.prototype.activate_field = function() {
      this.container.addClass("chzn-container-active");
      this.active_field = true;
      this.search_field.val(this.search_field.val());
      return this.search_field.focus();
    };

    Chosen.prototype.test_active_click = function(evt) {
      if ($(evt.target).parents('#' + this.container_id).length) {
        return this.active_field = true;
      } else {
        return this.close_field();
      }
    };

    Chosen.prototype.results_build = function() {
      var content, data, _i, _len, _ref;
      this.parsing = true;
      this.results_data = root.SelectParser.select_to_array(this.form_field);
      if (this.is_multiple && this.choices > 0) {
        this.search_choices.find("li.search-choice").remove();
        this.choices = 0;
      } else if (!this.is_multiple) {
        this.selected_item.addClass("chzn-default").find("span").text(this.default_text);
        if (this.disable_search || this.form_field.options.length <= this.disable_search_threshold) {
          this.container.addClass("chzn-container-single-nosearch");
        } else {
          this.container.removeClass("chzn-container-single-nosearch");
        }
      }
      content = '';
      _ref = this.results_data;
      for (_i = 0, _len = _ref.length; _i < _len; _i++) {
        data = _ref[_i];
        if (data.group) {
          content += this.result_add_group(data);
        } else if (!data.empty) {
          content += this.result_add_option(data);
          if (data.selected && this.is_multiple) {
            this.choice_build(data);
          } else if (data.selected && !this.is_multiple) {
            this.selected_item.removeClass("chzn-default").find("span").text(data.text);
            if (this.allow_single_deselect) {
              this.single_deselect_control_build();
            }
          }
        }
      }
      this.search_field_disabled();
      this.show_search_field_default();
      this.search_field_scale();
      this.search_results.html(content);
      return this.parsing = false;
    };

    Chosen.prototype.result_add_group = function(group) {
      if (!group.disabled) {
        group.dom_id = this.container_id + "_g_" + group.array_index;
        return '<li id="' + group.dom_id + '" class="group-result">' + $("<div />").text(group.label).html() + '</li>';
      } else {
        return "";
      }
    };

    Chosen.prototype.result_do_highlight = function(el) {
      var high_bottom, high_top, maxHeight, visible_bottom, visible_top;
      if (el.length) {
        this.result_clear_highlight();
        this.result_highlight = el;
        this.result_highlight.addClass("highlighted");
        maxHeight = parseInt(this.search_results.css("maxHeight"), 10);
        visible_top = this.search_results.scrollTop();
        visible_bottom = maxHeight + visible_top;
        high_top = this.result_highlight.position().top + this.search_results.scrollTop();
        high_bottom = high_top + this.result_highlight.outerHeight();
        if (high_bottom >= visible_bottom) {
          return this.search_results.scrollTop((high_bottom - maxHeight) > 0 ? high_bottom - maxHeight : 0);
        } else if (high_top < visible_top) {
          return this.search_results.scrollTop(high_top);
        }
      }
    };

    Chosen.prototype.result_clear_highlight = function() {
      if (this.result_highlight) {
        this.result_highlight.removeClass("highlighted");
      }
      return this.result_highlight = null;
    };

    Chosen.prototype.results_show = function() {
      var dd_top;
      if (!this.is_multiple) {
        this.selected_item.addClass("chzn-single-with-drop");
        if (this.result_single_selected) {
          this.result_do_highlight(this.result_single_selected);
        }
      } else if (this.max_selected_options <= this.choices) {
        this.form_field_jq.trigger("liszt:maxselected", {
          chosen: this
        });
        return false;
      }
      dd_top = this.is_multiple ? this.container.height() : this.container.height() - 1;
      this.form_field_jq.trigger("liszt:showing_dropdown", {
        chosen: this
      });
      this.dropdown.css({
        "top": dd_top + "px",
        "left": 0
      });
      this.results_showing = true;
      this.search_field.focus();
      this.search_field.val(this.search_field.val());
      return this.winnow_results();
    };

    Chosen.prototype.results_hide = function() {
      if (!this.is_multiple) {
        this.selected_item.removeClass("chzn-single-with-drop");
      }
      this.result_clear_highlight();
      this.form_field_jq.trigger("liszt:hiding_dropdown", {
        chosen: this
      });
      this.dropdown.css({
        "left": "-9000px"
      });
      return this.results_showing = false;
    };

    Chosen.prototype.set_tab_index = function(el) {
      var ti;
      if (this.form_field_jq.attr("tabindex")) {
        ti = this.form_field_jq.attr("tabindex");
        this.form_field_jq.attr("tabindex", -1);
        return this.search_field.attr("tabindex", ti);
      }
    };

    Chosen.prototype.show_search_field_default = function() {
      if (this.is_multiple && this.choices < 1 && !this.active_field) {
        this.search_field.val(this.default_text);
        return this.search_field.addClass("default");
      } else {
        this.search_field.val("");
        return this.search_field.removeClass("default");
      }
    };

    Chosen.prototype.search_results_mouseup = function(evt) {
      var target;
      target = $(evt.target).hasClass("active-result") ? $(evt.target) : $(evt.target).parents(".active-result").first();
      if (target.length) {
        this.result_highlight = target;
        this.result_select(evt);
        return this.search_field.focus();
      }
    };

    Chosen.prototype.search_results_mouseover = function(evt) {
      var target;
      target = $(evt.target).hasClass("active-result") ? $(evt.target) : $(evt.target).parents(".active-result").first();
      if (target) {
        return this.result_do_highlight(target);
      }
    };

    Chosen.prototype.search_results_mouseout = function(evt) {
      if ($(evt.target).hasClass("active-result" || $(evt.target).parents('.active-result').first())) {
        return this.result_clear_highlight();
      }
    };

    Chosen.prototype.choices_click = function(evt) {
      evt.preventDefault();
      if (this.active_field && !($(evt.target).hasClass("search-choice" || $(evt.target).parents('.search-choice').first)) && !this.results_showing) {
        return this.results_show();
      }
    };

    Chosen.prototype.choice_build = function(item) {
      var choice_id, html, link,
        _this = this;
      if (this.is_multiple && this.max_selected_options <= this.choices) {
        this.form_field_jq.trigger("liszt:maxselected", {
          chosen: this
        });
        return false;
      }
      choice_id = this.container_id + "_c_" + item.array_index;
      this.choices += 1;
      if (item.disabled) {
        html = '<li class="search-choice search-choice-disabled" id="' + choice_id + '"><span>' + item.html + '</span></li>';
      } else {
        html = '<li class="search-choice" id="' + choice_id + '"><span>' + item.html + '</span><a href="javascript:void(0)" class="search-choice-close" rel="' + item.array_index + '"></a></li>';
      }
      this.search_container.before(html);
      link = $('#' + choice_id).find("a").first();
      return link.click(function(evt) {
        return _this.choice_destroy_link_click(evt);
      });
    };

    Chosen.prototype.choice_destroy_link_click = function(evt) {
      evt.preventDefault();
      if (!this.is_disabled) {
        this.pending_destroy_click = true;
        return this.choice_destroy($(evt.target));
      } else {
        return evt.stopPropagation;
      }
    };

    Chosen.prototype.choice_destroy = function(link) {
      if (this.result_deselect(link.attr("rel"))) {
        this.choices -= 1;
        this.show_search_field_default();
        if (this.is_multiple && this.choices > 0 && this.search_field.val().length < 1) {
          this.results_hide();
        }
        link.parents('li').first().remove();
        return this.search_field_scale();
      }
    };

    Chosen.prototype.results_reset = function() {
      this.form_field.options[0].selected = true;
      this.selected_item.find("span").text(this.default_text);
      if (!this.is_multiple) {
        this.selected_item.addClass("chzn-default");
      }
      this.show_search_field_default();
      this.results_reset_cleanup();
      this.form_field_jq.trigger("change");
      if (this.active_field) {
        return this.results_hide();
      }
    };

    Chosen.prototype.results_reset_cleanup = function() {
      this.current_value = this.form_field_jq.val();
      return this.selected_item.find("abbr").remove();
    };

    Chosen.prototype.result_select = function(evt) {
      var high, high_id, item, position;
      if (this.result_highlight) {
        high = this.result_highlight;
        high_id = high.attr("id");
        this.result_clear_highlight();
        if (this.is_multiple) {
          this.result_deactivate(high);
        } else {
          this.search_results.find(".result-selected").removeClass("result-selected");
          this.result_single_selected = high;
          this.selected_item.removeClass("chzn-default");
        }
        high.addClass("result-selected");
        position = high_id.substr(high_id.lastIndexOf("_") + 1);
        item = this.results_data[position];
        item.selected = true;
        this.form_field.options[item.options_index].selected = true;
        if (this.is_multiple) {
          this.choice_build(item);
        } else {
          this.selected_item.find("span").first().text(item.text);
          if (this.allow_single_deselect) {
            this.single_deselect_control_build();
          }
        }
        if (!((evt.metaKey || evt.ctrlKey) && this.is_multiple)) {
          this.results_hide();
        }
        this.search_field.val("");
        if (this.is_multiple || this.form_field_jq.val() !== this.current_value) {
          this.form_field_jq.trigger("change", {
            'selected': this.form_field.options[item.options_index].value
          });
        }
        this.current_value = this.form_field_jq.val();
        return this.search_field_scale();
      }
    };

    Chosen.prototype.result_activate = function(el) {
      return el.addClass("active-result");
    };

    Chosen.prototype.result_deactivate = function(el) {
      return el.removeClass("active-result");
    };

    Chosen.prototype.result_deselect = function(pos) {
      var result, result_data;
      result_data = this.results_data[pos];
      if (!this.form_field.options[result_data.options_index].disabled) {
        result_data.selected = false;
        this.form_field.options[result_data.options_index].selected = false;
        result = $("#" + this.container_id + "_o_" + pos);
        result.removeClass("result-selected").addClass("active-result").show();
        this.result_clear_highlight();
        this.winnow_results();
        this.form_field_jq.trigger("change", {
          deselected: this.form_field.options[result_data.options_index].value
        });
        this.search_field_scale();
        return true;
      } else {
        return false;
      }
    };

    Chosen.prototype.single_deselect_control_build = function() {
      if (this.allow_single_deselect && this.selected_item.find("abbr").length < 1) {
        return this.selected_item.find("span").first().after("<abbr class=\"search-choice-close\"></abbr>");
      }
    };

    Chosen.prototype.winnow_results = function() {
      var found, option, part, parts, regex, regexAnchor, result, result_id, results, searchText, startpos, text, zregex, _i, _j, _len, _len1, _ref;
      this.no_results_clear();
      results = 0;
      searchText = this.search_field.val() === this.default_text ? "" : $('<div/>').text($.trim(this.search_field.val())).html();
      regexAnchor = this.search_contains ? "" : "^";
      regex = new RegExp(regexAnchor + searchText.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, "\\$&"), 'i');
      zregex = new RegExp(searchText.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, "\\$&"), 'i');
      _ref = this.results_data;
      for (_i = 0, _len = _ref.length; _i < _len; _i++) {
        option = _ref[_i];
        if (!option.disabled && !option.empty) {
          if (option.group) {
            $('#' + option.dom_id).css('display', 'none');
          } else if (!(this.is_multiple && option.selected)) {
            found = false;
            result_id = option.dom_id;
            result = $("#" + result_id);
            if (regex.test(option.html)) {
              found = true;
              results += 1;
            } else if (this.enable_split_word_search && (option.html.indexOf(" ") >= 0 || option.html.indexOf("[") === 0)) {
              parts = option.html.replace(/\[|\]/g, "").split(" ");
              if (parts.length) {
                for (_j = 0, _len1 = parts.length; _j < _len1; _j++) {
                  part = parts[_j];
                  if (regex.test(part)) {
                    found = true;
                    results += 1;
                  }
                }
              }
            }
            if (found) {
              if (searchText.length) {
                startpos = option.html.search(zregex);
                text = option.html.substr(0, startpos + searchText.length) + '</em>' + option.html.substr(startpos + searchText.length);
                text = text.substr(0, startpos) + '<em>' + text.substr(startpos);
              } else {
                text = option.html;
              }
              result.html(text);
              this.result_activate(result);
              if (option.group_array_index != null) {
                $("#" + this.results_data[option.group_array_index].dom_id).css('display', 'list-item');
              }
            } else {
              if (this.result_highlight && result_id === this.result_highlight.attr('id')) {
                this.result_clear_highlight();
              }
              this.result_deactivate(result);
            }
          }
        }
      }
      if (results < 1 && searchText.length) {
        return this.no_results(searchText);
      } else {
        return this.winnow_results_set_highlight();
      }
    };

    Chosen.prototype.winnow_results_clear = function() {
      var li, lis, _i, _len, _results;
      this.search_field.val("");
      lis = this.search_results.find("li");
      _results = [];
      for (_i = 0, _len = lis.length; _i < _len; _i++) {
        li = lis[_i];
        li = $(li);
        if (li.hasClass("group-result")) {
          _results.push(li.css('display', 'auto'));
        } else if (!this.is_multiple || !li.hasClass("result-selected")) {
          _results.push(this.result_activate(li));
        } else {
          _results.push(void 0);
        }
      }
      return _results;
    };

    Chosen.prototype.winnow_results_set_highlight = function() {
      var do_high, selected_results;
      if (!this.result_highlight) {
        selected_results = !this.is_multiple ? this.search_results.find(".result-selected.active-result") : [];
        do_high = selected_results.length ? selected_results.first() : this.search_results.find(".active-result").first();
        if (do_high != null) {
          return this.result_do_highlight(do_high);
        }
      }
    };

    Chosen.prototype.no_results = function(terms) {
      var no_results_html;
      no_results_html = $('<li class="no-results">' + this.results_none_found + ' "<span></span>"</li>');
      no_results_html.find("span").first().html(terms);
      return this.search_results.append(no_results_html);
    };

    Chosen.prototype.no_results_clear = function() {
      return this.search_results.find(".no-results").remove();
    };

    Chosen.prototype.keydown_arrow = function() {
      var first_active, next_sib;
      if (!this.result_highlight) {
        first_active = this.search_results.find("li.active-result").first();
        if (first_active) {
          this.result_do_highlight($(first_active));
        }
      } else if (this.results_showing) {
        next_sib = this.result_highlight.nextAll("li.active-result").first();
        if (next_sib) {
          this.result_do_highlight(next_sib);
        }
      }
      if (!this.results_showing) {
        return this.results_show();
      }
    };

    Chosen.prototype.keyup_arrow = function() {
      var prev_sibs;
      if (!this.results_showing && !this.is_multiple) {
        return this.results_show();
      } else if (this.result_highlight) {
        prev_sibs = this.result_highlight.prevAll("li.active-result");
        if (prev_sibs.length) {
          return this.result_do_highlight(prev_sibs.first());
        } else {
          if (this.choices > 0) {
            this.results_hide();
          }
          return this.result_clear_highlight();
        }
      }
    };

    Chosen.prototype.keydown_backstroke = function() {
      var next_available_destroy;
      if (this.pending_backstroke) {
        this.choice_destroy(this.pending_backstroke.find("a").first());
        return this.clear_backstroke();
      } else {
        next_available_destroy = this.search_container.siblings("li.search-choice").last();
        if (next_available_destroy.length && !next_available_destroy.hasClass("search-choice-disabled")) {
          this.pending_backstroke = next_available_destroy;
          if (this.single_backstroke_delete) {
            return this.keydown_backstroke();
          } else {
            return this.pending_backstroke.addClass("search-choice-focus");
          }
        }
      }
    };

    Chosen.prototype.clear_backstroke = function() {
      if (this.pending_backstroke) {
        this.pending_backstroke.removeClass("search-choice-focus");
      }
      return this.pending_backstroke = null;
    };

    Chosen.prototype.keydown_checker = function(evt) {
      var stroke, _ref;
      stroke = (_ref = evt.which) != null ? _ref : evt.keyCode;
      this.search_field_scale();
      if (stroke !== 8 && this.pending_backstroke) {
        this.clear_backstroke();
      }
      switch (stroke) {
        case 8:
          this.backstroke_length = this.search_field.val().length;
          break;
        case 9:
          if (this.results_showing && !this.is_multiple) {
            this.result_select(evt);
          }
          this.mouse_on_container = false;
          break;
        case 13:
          evt.preventDefault();
          break;
        case 38:
          evt.preventDefault();
          this.keyup_arrow();
          break;
        case 40:
          this.keydown_arrow();
          break;
      }
    };

    Chosen.prototype.search_field_scale = function() {
      var dd_top, div, h, style, style_block, styles, w, _i, _len;
      if (this.is_multiple) {
        h = 0;
        w = 0;
        style_block = "position:absolute; left: -1000px; top: -1000px; display:none;";
        styles = ['font-size', 'font-style', 'font-weight', 'font-family', 'line-height', 'text-transform', 'letter-spacing'];
        for (_i = 0, _len = styles.length; _i < _len; _i++) {
          style = styles[_i];
          style_block += style + ":" + this.search_field.css(style) + ";";
        }
        div = $('<div />', {
          'style': style_block
        });
        div.text(this.search_field.val());
        $('body').append(div);
        w = div.width() + 25;
        div.remove();
        if (w > this.f_width - 10) {
          w = this.f_width - 10;
        }
        this.search_field.css({
          'width': w + 'px'
        });
        dd_top = this.container.height();
        return this.dropdown.css({
          "top": dd_top + "px"
        });
      }
    };

    Chosen.prototype.generate_random_id = function() {
      var string;
      string = "sel" + this.generate_random_char() + this.generate_random_char() + this.generate_random_char();
      while ($("#" + string).length > 0) {
        string += this.generate_random_char();
      }
      return string;
    };

    return Chosen;

  })(AbstractChosen);

  root.Chosen = Chosen;

  get_side_border_padding = function(elmt) {
    var side_border_padding;
    return side_border_padding = elmt.outerWidth() - elmt.width();
  };

  root.get_side_border_padding = get_side_border_padding;

}).call(this);/*!
 * jQuery Cookie Plugin v1.4.0
 * https://github.com/carhartl/jquery-cookie
 *
 * Copyright 2013 Klaus Hartl
 * Released under the MIT license
 */
(function (factory) {
	if (typeof define === 'function' && define.amd) {
		// AMD. Register as anonymous module.
		define(['jquery'], factory);
	} else {
		// Browser globals.
		factory(jQuery);
	}
}(function ($) {

	var pluses = /\+/g;

	function encode(s) {
		return config.raw ? s : encodeURIComponent(s);
	}

	function decode(s) {
		return config.raw ? s : decodeURIComponent(s);
	}

	function stringifyCookieValue(value) {
		return encode(config.json ? JSON.stringify(value) : String(value));
	}

	function parseCookieValue(s) {
		if (s.indexOf('"') === 0) {
			// This is a quoted cookie as according to RFC2068, unescape...
			s = s.slice(1, -1).replace(/\\"/g, '"').replace(/\\\\/g, '\\');
		}

		try {
			// Replace server-side written pluses with spaces.
			// If we can't decode the cookie, ignore it, it's unusable.
			s = decodeURIComponent(s.replace(pluses, ' '));
		} catch(e) {
			return;
		}

		try {
			// If we can't parse the cookie, ignore it, it's unusable.
			return config.json ? JSON.parse(s) : s;
		} catch(e) {}
	}

	function read(s, converter) {
		var value = config.raw ? s : parseCookieValue(s);
		return $.isFunction(converter) ? converter(value) : value;
	}

	var config = $.cookie = function (key, value, options) {

		// Write
		if (value !== undefined && !$.isFunction(value)) {
			options = $.extend({}, config.defaults, options);

			if (typeof options.expires === 'number') {
				var days = options.expires, t = options.expires = new Date();
				t.setDate(t.getDate() + days);
			}

			return (document.cookie = [
				encode(key), '=', stringifyCookieValue(value),
				options.expires ? '; expires=' + options.expires.toUTCString() : '', // use expires attribute, max-age is not supported by IE
				options.path    ? '; path=' + options.path : '',
				options.domain  ? '; domain=' + options.domain : '',
				options.secure  ? '; secure' : ''
			].join(''));
		}

		// Read

		var result = key ? undefined : {};

		// To prevent the for loop in the first place assign an empty array
		// in case there are no cookies at all. Also prevents odd result when
		// calling $.cookie().
		var cookies = document.cookie ? document.cookie.split('; ') : [];

		for (var i = 0, l = cookies.length; i < l; i++) {
			var parts = cookies[i].split('=');
			var name = decode(parts.shift());
			var cookie = parts.join('=');

			if (key && key === name) {
				// If second argument (value) is a function it's a converter...
				result = read(cookie, value);
				break;
			}

			// Prevent storing a cookie that we couldn't decode.
			if (!key && (cookie = read(cookie)) !== undefined) {
				result[name] = cookie;
			}
		}

		return result;
	};

	config.defaults = {};

	$.removeCookie = function (key, options) {
		if ($.cookie(key) !== undefined) {
			// Must not alter options, thus extending a fresh object...
			$.cookie(key, '', $.extend({}, options, { expires: -1 }));
			return true;
		}
		return false;
	};

}));
/*
 * jQuery plugin: fieldSelection - v0.1.0 - last change: 2006-12-16
 * (c) 2006 Alex Brem <alex@0xab.cd> - http://blog.0xab.cd
 */

(function() {

	var fieldSelection = {

		getSelection: function() {

			var e = this.jquery ? this[0] : this;

			return (

				/* mozilla / dom 3.0 */
				('selectionStart' in e && function() {
					var l = e.selectionEnd - e.selectionStart;
					return { start: e.selectionStart, end: e.selectionEnd, length: l, text: e.value.substr(e.selectionStart, l) };
				}) ||

				/* exploder */
				(document.selection && function() {

					e.focus();

					var r = document.selection.createRange();
					if (r == null) {
						return { start: 0, end: e.value.length, length: 0 }
					}

					var re = e.createTextRange();
					var rc = re.duplicate();
					re.moveToBookmark(r.getBookmark());
					rc.setEndPoint('EndToStart', re);

					return { start: rc.text.length, end: rc.text.length + r.text.length, length: r.text.length, text: r.text };
				}) ||

				/* browser not supported */
				function() {
					return { start: 0, end: e.value.length, length: 0 };
				}

			)();

		},

		replaceSelection: function() {

			var e = this.jquery ? this[0] : this;
			var text = arguments[0] || '';

			return (

				/* mozilla / dom 3.0 */
				('selectionStart' in e && function() {
					e.value = e.value.substr(0, e.selectionStart) + text + e.value.substr(e.selectionEnd, e.value.length);
					return this;
				}) ||

				/* exploder */
				(document.selection && function() {
					e.focus();
					document.selection.createRange().text = text;
					return this;
				}) ||

				/* browser not supported */
				function() {
					e.value += text;
					return this;
				}

			)();

		}

	};

	jQuery.each(fieldSelection, function(i) { jQuery.fn[i] = this; });

})();
/*
 * Lazy Load - jQuery plugin for lazy loading images
 *
 * Copyright (c) 2007-2013 Mika Tuupola
 *
 * Licensed under the MIT license:
 *   http://www.opensource.org/licenses/mit-license.php
 *
 * Project home:
 *   http://www.appelsiini.net/projects/lazyload
 *
 * Version:  1.9.1
 *
 */

(function($, window, document, undefined) {
    var $window = $(window);

    $.fn.lazyload = function(options) {
        var elements = this;
        var $container;
        var settings = {
            threshold       : 0,
            failure_limit   : 0,
            event           : "scroll",
            effect          : "show",
            container       : window,
            data_attribute  : "original",
            skip_invisible  : true,
            appear          : null,
            load            : null,
            placeholder     : "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsQAAA7EAZUrDhsAAAANSURBVBhXYzh8+PB/AAffA0nNPuCLAAAAAElFTkSuQmCC"
        };

        function update() {
            var counter = 0;

            elements.each(function() {
                var $this = $(this);
                if (settings.skip_invisible && !$this.is(":visible")) {
                    return;
                }
                if ($.abovethetop(this, settings) ||
                    $.leftofbegin(this, settings)) {
                        /* Nothing. */
                } else if (!$.belowthefold(this, settings) &&
                    !$.rightoffold(this, settings)) {
                        $this.trigger("appear");
                        /* if we found an image we'll load, reset the counter */
                        counter = 0;
                } else {
                    if (++counter > settings.failure_limit) {
                        return false;
                    }
                }
            });

        }

        if(options) {
            /* Maintain BC for a couple of versions. */
            if (undefined !== options.failurelimit) {
                options.failure_limit = options.failurelimit;
                delete options.failurelimit;
            }
            if (undefined !== options.effectspeed) {
                options.effect_speed = options.effectspeed;
                delete options.effectspeed;
            }

            $.extend(settings, options);
        }

        /* Cache container as jQuery as object. */
        $container = (settings.container === undefined ||
                      settings.container === window) ? $window : $(settings.container);

        /* Fire one scroll event per scroll. Not one scroll event per image. */
        if (0 === settings.event.indexOf("scroll")) {
            $container.bind(settings.event, function() {
                return update();
            });
        }

        this.each(function() {
            var self = this;
            var $self = $(self);

            self.loaded = false;

            /* If no src attribute given use data:uri. */
            if ($self.attr("src") === undefined || $self.attr("src") === false) {
                if ($self.is("img")) {
                    $self.attr("src", settings.placeholder);
                }
            }

            /* When appear is triggered load original image. */
            $self.one("appear", function() {
                if (!this.loaded) {
                    if (settings.appear) {
                        var elements_left = elements.length;
                        settings.appear.call(self, elements_left, settings);
                    }
                    $("<img />")
                        .bind("load", function() {

                            var original = $self.attr("data-" + settings.data_attribute);
                            $self.hide();
                            if ($self.is("img")) {
                                $self.attr("src", original);
                            } else {
                                $self.css("background-image", "url('" + original + "')");
                            }
                            $self[settings.effect](settings.effect_speed);

                            self.loaded = true;

                            /* Remove image from array so it is not looped next time. */
                            var temp = $.grep(elements, function(element) {
                                return !element.loaded;
                            });
                            elements = $(temp);

                            if (settings.load) {
                                var elements_left = elements.length;
                                settings.load.call(self, elements_left, settings);
                            }
                        })
                        .attr("src", $self.attr("data-" + settings.data_attribute));
                }
            });

            /* When wanted event is triggered load original image */
            /* by triggering appear.                              */
            if (0 !== settings.event.indexOf("scroll")) {
                $self.bind(settings.event, function() {
                    if (!self.loaded) {
                        $self.trigger("appear");
                    }
                });
            }
        });

        /* Check if something appears when window is resized. */
        $window.bind("resize", function() {
            update();
        });

        /* With IOS5 force loading images when navigating with back button. */
        /* Non optimal workaround. */
        if ((/(?:iphone|ipod|ipad).*os 5/gi).test(navigator.appVersion)) {
            $window.bind("pageshow", function(event) {
                if (event.originalEvent && event.originalEvent.persisted) {
                    elements.each(function() {
                        $(this).trigger("appear");
                    });
                }
            });
        }

        /* Force initial check if images should appear. */
        $(document).ready(function() {
            update();
        });

        return this;
    };

    /* Convenience methods in jQuery namespace.           */
    /* Use as  $.belowthefold(element, {threshold : 100, container : window}) */

    $.belowthefold = function(element, settings) {
        var fold;

        if (settings.container === undefined || settings.container === window) {
            fold = (window.innerHeight ? window.innerHeight : $window.height()) + $window.scrollTop();
        } else {
            fold = $(settings.container).offset().top + $(settings.container).height();
        }

        return fold <= $(element).offset().top - settings.threshold;
    };

    $.rightoffold = function(element, settings) {
        var fold;

        if (settings.container === undefined || settings.container === window) {
            fold = $window.width() + $window.scrollLeft();
        } else {
            fold = $(settings.container).offset().left + $(settings.container).width();
        }

        return fold <= $(element).offset().left - settings.threshold;
    };

    $.abovethetop = function(element, settings) {
        var fold;

        if (settings.container === undefined || settings.container === window) {
            fold = $window.scrollTop();
        } else {
            fold = $(settings.container).offset().top;
        }

        return fold >= $(element).offset().top + settings.threshold  + $(element).height();
    };

    $.leftofbegin = function(element, settings) {
        var fold;

        if (settings.container === undefined || settings.container === window) {
            fold = $window.scrollLeft();
        } else {
            fold = $(settings.container).offset().left;
        }

        return fold >= $(element).offset().left + settings.threshold + $(element).width();
    };

    $.inviewport = function(element, settings) {
         return !$.rightoffold(element, settings) && !$.leftofbegin(element, settings) &&
                !$.belowthefold(element, settings) && !$.abovethetop(element, settings);
     };

    /* Custom selectors for your convenience.   */
    /* Use as $("img:below-the-fold").something() or */
    /* $("img").filter(":below-the-fold").something() which is faster */

    $.extend($.expr[":"], {
        "below-the-fold" : function(a) { return $.belowthefold(a, {threshold : 0}); },
        "above-the-top"  : function(a) { return !$.belowthefold(a, {threshold : 0}); },
        "right-of-screen": function(a) { return $.rightoffold(a, {threshold : 0}); },
        "left-of-screen" : function(a) { return !$.rightoffold(a, {threshold : 0}); },
        "in-viewport"    : function(a) { return $.inviewport(a, {threshold : 0}); },
        /* Maintain BC for couple of versions. */
        "above-the-fold" : function(a) { return !$.belowthefold(a, {threshold : 0}); },
        "right-of-fold"  : function(a) { return $.rightoffold(a, {threshold : 0}); },
        "left-of-fold"   : function(a) { return !$.rightoffold(a, {threshold : 0}); }
    });

})(jQuery, window, document);
/*! jCarousel - v0.3.0 - 2013-11-22
* http://sorgalla.com/jcarousel
* Copyright (c) 2013 Jan Sorgalla; Licensed MIT */
(function(t){"use strict";var i=t.jCarousel={};i.version="0.3.0";var s=/^([+\-]=)?(.+)$/;i.parseTarget=function(t){var i=!1,e="object"!=typeof t?s.exec(t):null;return e?(t=parseInt(e[2],10)||0,e[1]&&(i=!0,"-="===e[1]&&(t*=-1))):"object"!=typeof t&&(t=parseInt(t,10)||0),{target:t,relative:i}},i.detectCarousel=function(t){for(var i;t.length>0;){if(i=t.filter("[data-jcarousel]"),i.length>0)return i;if(i=t.find("[data-jcarousel]"),i.length>0)return i;t=t.parent()}return null},i.base=function(s){return{version:i.version,_options:{},_element:null,_carousel:null,_init:t.noop,_create:t.noop,_destroy:t.noop,_reload:t.noop,create:function(){return this._element.attr("data-"+s.toLowerCase(),!0).data(s,this),!1===this._trigger("create")?this:(this._create(),this._trigger("createend"),this)},destroy:function(){return!1===this._trigger("destroy")?this:(this._destroy(),this._trigger("destroyend"),this._element.removeData(s).removeAttr("data-"+s.toLowerCase()),this)},reload:function(t){return!1===this._trigger("reload")?this:(t&&this.options(t),this._reload(),this._trigger("reloadend"),this)},element:function(){return this._element},options:function(i,s){if(0===arguments.length)return t.extend({},this._options);if("string"==typeof i){if(s===void 0)return this._options[i]===void 0?null:this._options[i];this._options[i]=s}else this._options=t.extend({},this._options,i);return this},carousel:function(){return this._carousel||(this._carousel=i.detectCarousel(this.options("carousel")||this._element),this._carousel||t.error('Could not detect carousel for plugin "'+s+'"')),this._carousel},_trigger:function(i,e,r){var n,o=!1;return r=[this].concat(r||[]),(e||this._element).each(function(){n=t.Event((s+":"+i).toLowerCase()),t(this).trigger(n,r),n.isDefaultPrevented()&&(o=!0)}),!o}}},i.plugin=function(s,e){var r=t[s]=function(i,s){this._element=t(i),this.options(s),this._init(),this.create()};return r.fn=r.prototype=t.extend({},i.base(s),e),t.fn[s]=function(i){var e=Array.prototype.slice.call(arguments,1),n=this;return"string"==typeof i?this.each(function(){var r=t(this).data(s);if(!r)return t.error("Cannot call methods on "+s+" prior to initialization; "+'attempted to call method "'+i+'"');if(!t.isFunction(r[i])||"_"===i.charAt(0))return t.error('No such method "'+i+'" for '+s+" instance");var o=r[i].apply(r,e);return o!==r&&o!==void 0?(n=o,!1):void 0}):this.each(function(){var e=t(this).data(s);e instanceof r?e.reload(i):new r(this,i)}),n},r}})(jQuery),function(t,i){"use strict";var s=function(t){return parseFloat(t)||0};t.jCarousel.plugin("jcarousel",{animating:!1,tail:0,inTail:!1,resizeTimer:null,lt:null,vertical:!1,rtl:!1,circular:!1,underflow:!1,relative:!1,_options:{list:function(){return this.element().children().eq(0)},items:function(){return this.list().children()},animation:400,transitions:!1,wrap:null,vertical:null,rtl:null,center:!1},_list:null,_items:null,_target:null,_first:null,_last:null,_visible:null,_fullyvisible:null,_init:function(){var t=this;return this.onWindowResize=function(){t.resizeTimer&&clearTimeout(t.resizeTimer),t.resizeTimer=setTimeout(function(){t.reload()},100)},this},_create:function(){this._reload(),t(i).on("resize.jcarousel",this.onWindowResize)},_destroy:function(){t(i).off("resize.jcarousel",this.onWindowResize)},_reload:function(){this.vertical=this.options("vertical"),null==this.vertical&&(this.vertical=this.list().height()>this.list().width()),this.rtl=this.options("rtl"),null==this.rtl&&(this.rtl=function(i){if("rtl"===(""+i.attr("dir")).toLowerCase())return!0;var s=!1;return i.parents("[dir]").each(function(){return/rtl/i.test(t(this).attr("dir"))?(s=!0,!1):void 0}),s}(this._element)),this.lt=this.vertical?"top":"left",this.relative="relative"===this.list().css("position"),this._list=null,this._items=null;var i=this._target&&this.index(this._target)>=0?this._target:this.closest();this.circular="circular"===this.options("wrap"),this.underflow=!1;var s={left:0,top:0};return i.length>0&&(this._prepare(i),this.list().find("[data-jcarousel-clone]").remove(),this._items=null,this.underflow=this._fullyvisible.length>=this.items().length,this.circular=this.circular&&!this.underflow,s[this.lt]=this._position(i)+"px"),this.move(s),this},list:function(){if(null===this._list){var i=this.options("list");this._list=t.isFunction(i)?i.call(this):this._element.find(i)}return this._list},items:function(){if(null===this._items){var i=this.options("items");this._items=(t.isFunction(i)?i.call(this):this.list().find(i)).not("[data-jcarousel-clone]")}return this._items},index:function(t){return this.items().index(t)},closest:function(){var i,e=this,r=this.list().position()[this.lt],n=t(),o=!1,l=this.vertical?"bottom":this.rtl&&!this.relative?"left":"right";return this.rtl&&this.relative&&!this.vertical&&(r+=this.list().width()-this.clipping()),this.items().each(function(){if(n=t(this),o)return!1;var a=e.dimension(n);if(r+=a,r>=0){if(i=a-s(n.css("margin-"+l)),!(0>=Math.abs(r)-a+i/2))return!1;o=!0}}),n},target:function(){return this._target},first:function(){return this._first},last:function(){return this._last},visible:function(){return this._visible},fullyvisible:function(){return this._fullyvisible},hasNext:function(){if(!1===this._trigger("hasnext"))return!0;var t=this.options("wrap"),i=this.items().length-1;return i>=0&&(t&&"first"!==t||i>this.index(this._last)||this.tail&&!this.inTail)?!0:!1},hasPrev:function(){if(!1===this._trigger("hasprev"))return!0;var t=this.options("wrap");return this.items().length>0&&(t&&"last"!==t||this.index(this._first)>0||this.tail&&this.inTail)?!0:!1},clipping:function(){return this._element["inner"+(this.vertical?"Height":"Width")]()},dimension:function(t){return t["outer"+(this.vertical?"Height":"Width")](!0)},scroll:function(i,s,e){if(this.animating)return this;if(!1===this._trigger("scroll",null,[i,s]))return this;t.isFunction(s)&&(e=s,s=!0);var r=t.jCarousel.parseTarget(i);if(r.relative){var n,o,l,a,h,u,c,f,d=this.items().length-1,_=Math.abs(r.target),p=this.options("wrap");if(r.target>0){var v=this.index(this._last);if(v>=d&&this.tail)this.inTail?"both"===p||"last"===p?this._scroll(0,s,e):t.isFunction(e)&&e.call(this,!1):this._scrollTail(s,e);else if(n=this.index(this._target),this.underflow&&n===d&&("circular"===p||"both"===p||"last"===p)||!this.underflow&&v===d&&("both"===p||"last"===p))this._scroll(0,s,e);else if(l=n+_,this.circular&&l>d){for(f=d,h=this.items().get(-1);l>f++;)h=this.items().eq(0),u=this._visible.index(h)>=0,u&&h.after(h.clone(!0).attr("data-jcarousel-clone",!0)),this.list().append(h),u||(c={},c[this.lt]=this.dimension(h),this.moveBy(c)),this._items=null;this._scroll(h,s,e)}else this._scroll(Math.min(l,d),s,e)}else if(this.inTail)this._scroll(Math.max(this.index(this._first)-_+1,0),s,e);else if(o=this.index(this._first),n=this.index(this._target),a=this.underflow?n:o,l=a-_,0>=a&&(this.underflow&&"circular"===p||"both"===p||"first"===p))this._scroll(d,s,e);else if(this.circular&&0>l){for(f=l,h=this.items().get(0);0>f++;){h=this.items().eq(-1),u=this._visible.index(h)>=0,u&&h.after(h.clone(!0).attr("data-jcarousel-clone",!0)),this.list().prepend(h),this._items=null;var g=this.dimension(h);c={},c[this.lt]=-g,this.moveBy(c)}this._scroll(h,s,e)}else this._scroll(Math.max(l,0),s,e)}else this._scroll(r.target,s,e);return this._trigger("scrollend"),this},moveBy:function(t,i){var e=this.list().position(),r=1,n=0;return this.rtl&&!this.vertical&&(r=-1,this.relative&&(n=this.list().width()-this.clipping())),t.left&&(t.left=e.left+n+s(t.left)*r+"px"),t.top&&(t.top=e.top+n+s(t.top)*r+"px"),this.move(t,i)},move:function(i,s){s=s||{};var e=this.options("transitions"),r=!!e,n=!!e.transforms,o=!!e.transforms3d,l=s.duration||0,a=this.list();if(!r&&l>0)return a.animate(i,s),void 0;var h=s.complete||t.noop,u={};if(r){var c=a.css(["transitionDuration","transitionTimingFunction","transitionProperty"]),f=h;h=function(){t(this).css(c),f.call(this)},u={transitionDuration:(l>0?l/1e3:0)+"s",transitionTimingFunction:e.easing||s.easing,transitionProperty:l>0?function(){return n||o?"all":i.left?"left":"top"}():"none",transform:"none"}}o?u.transform="translate3d("+(i.left||0)+","+(i.top||0)+",0)":n?u.transform="translate("+(i.left||0)+","+(i.top||0)+")":t.extend(u,i),r&&l>0&&a.one("transitionend webkitTransitionEnd oTransitionEnd otransitionend MSTransitionEnd",h),a.css(u),0>=l&&a.each(function(){h.call(this)})},_scroll:function(i,s,e){if(this.animating)return t.isFunction(e)&&e.call(this,!1),this;if("object"!=typeof i?i=this.items().eq(i):i.jquery===void 0&&(i=t(i)),0===i.length)return t.isFunction(e)&&e.call(this,!1),this;this.inTail=!1,this._prepare(i);var r=this._position(i),n=this.list().position()[this.lt];if(r===n)return t.isFunction(e)&&e.call(this,!1),this;var o={};return o[this.lt]=r+"px",this._animate(o,s,e),this},_scrollTail:function(i,s){if(this.animating||!this.tail)return t.isFunction(s)&&s.call(this,!1),this;var e=this.list().position()[this.lt];this.rtl&&this.relative&&!this.vertical&&(e+=this.list().width()-this.clipping()),this.rtl&&!this.vertical?e+=this.tail:e-=this.tail,this.inTail=!0;var r={};return r[this.lt]=e+"px",this._update({target:this._target.next(),fullyvisible:this._fullyvisible.slice(1).add(this._visible.last())}),this._animate(r,i,s),this},_animate:function(i,s,e){if(e=e||t.noop,!1===this._trigger("animate"))return e.call(this,!1),this;this.animating=!0;var r=this.options("animation"),n=t.proxy(function(){this.animating=!1;var t=this.list().find("[data-jcarousel-clone]");t.length>0&&(t.remove(),this._reload()),this._trigger("animateend"),e.call(this,!0)},this),o="object"==typeof r?t.extend({},r):{duration:r},l=o.complete||t.noop;return s===!1?o.duration=0:t.fx.speeds[o.duration]!==void 0&&(o.duration=t.fx.speeds[o.duration]),o.complete=function(){n(),l.call(this)},this.move(i,o),this},_prepare:function(i){var e,r,n,o,l=this.index(i),a=l,h=this.dimension(i),u=this.clipping(),c=this.vertical?"bottom":this.rtl?"left":"right",f=this.options("center"),d={target:i,first:i,last:i,visible:i,fullyvisible:u>=h?i:t()};if(f&&(h/=2,u/=2),u>h)for(;;){if(e=this.items().eq(++a),0===e.length){if(!this.circular)break;if(e=this.items().eq(0),i.get(0)===e.get(0))break;if(r=this._visible.index(e)>=0,r&&e.after(e.clone(!0).attr("data-jcarousel-clone",!0)),this.list().append(e),!r){var _={};_[this.lt]=this.dimension(e),this.moveBy(_)}this._items=null}if(o=this.dimension(e),0===o)break;if(h+=o,d.last=e,d.visible=d.visible.add(e),n=s(e.css("margin-"+c)),u>=h-n&&(d.fullyvisible=d.fullyvisible.add(e)),h>=u)break}if(!this.circular&&!f&&u>h)for(a=l;;){if(0>--a)break;if(e=this.items().eq(a),0===e.length)break;if(o=this.dimension(e),0===o)break;if(h+=o,d.first=e,d.visible=d.visible.add(e),n=s(e.css("margin-"+c)),u>=h-n&&(d.fullyvisible=d.fullyvisible.add(e)),h>=u)break}return this._update(d),this.tail=0,f||"circular"===this.options("wrap")||"custom"===this.options("wrap")||this.index(d.last)!==this.items().length-1||(h-=s(d.last.css("margin-"+c)),h>u&&(this.tail=h-u)),this},_position:function(t){var i=this._first,s=i.position()[this.lt],e=this.options("center"),r=e?this.clipping()/2-this.dimension(i)/2:0;return this.rtl&&!this.vertical?(s-=this.relative?this.list().width()-this.dimension(i):this.clipping()-this.dimension(i),s+=r):s-=r,!e&&(this.index(t)>this.index(i)||this.inTail)&&this.tail?(s=this.rtl&&!this.vertical?s-this.tail:s+this.tail,this.inTail=!0):this.inTail=!1,-s},_update:function(i){var s,e=this,r={target:this._target||t(),first:this._first||t(),last:this._last||t(),visible:this._visible||t(),fullyvisible:this._fullyvisible||t()},n=this.index(i.first||r.first)<this.index(r.first),o=function(s){var o=[],l=[];i[s].each(function(){0>r[s].index(this)&&o.push(this)}),r[s].each(function(){0>i[s].index(this)&&l.push(this)}),n?o=o.reverse():l=l.reverse(),e._trigger(s+"in",t(o)),e._trigger(s+"out",t(l)),e["_"+s]=i[s]};for(s in i)o(s);return this}})}(jQuery,window),function(t){"use strict";t.jcarousel.fn.scrollIntoView=function(i,s,e){var r,n=t.jCarousel.parseTarget(i),o=this.index(this._fullyvisible.first()),l=this.index(this._fullyvisible.last());if(r=n.relative?0>n.target?Math.max(0,o+n.target):l+n.target:"object"!=typeof n.target?n.target:this.index(n.target),o>r)return this.scroll(r,s,e);if(r>=o&&l>=r)return t.isFunction(e)&&e.call(this,!1),this;for(var a,h=this.items(),u=this.clipping(),c=this.vertical?"bottom":this.rtl?"left":"right",f=0;;){if(a=h.eq(r),0===a.length)break;if(f+=this.dimension(a),f>=u){var d=parseFloat(a.css("margin-"+c))||0;f-d!==u&&r++;break}if(0>=r)break;r--}return this.scroll(r,s,e)}}(jQuery),function(t){"use strict";t.jCarousel.plugin("jcarouselControl",{_options:{target:"+=1",event:"click",method:"scroll"},_active:null,_init:function(){this.onDestroy=t.proxy(function(){this._destroy(),this.carousel().one("jcarousel:createend",t.proxy(this._create,this))},this),this.onReload=t.proxy(this._reload,this),this.onEvent=t.proxy(function(i){i.preventDefault();var s=this.options("method");t.isFunction(s)?s.call(this):this.carousel().jcarousel(this.options("method"),this.options("target"))},this)},_create:function(){this.carousel().one("jcarousel:destroy",this.onDestroy).on("jcarousel:reloadend jcarousel:scrollend",this.onReload),this._element.on(this.options("event")+".jcarouselcontrol",this.onEvent),this._reload()},_destroy:function(){this._element.off(".jcarouselcontrol",this.onEvent),this.carousel().off("jcarousel:destroy",this.onDestroy).off("jcarousel:reloadend jcarousel:scrollend",this.onReload)},_reload:function(){var i,s=t.jCarousel.parseTarget(this.options("target")),e=this.carousel();if(s.relative)i=e.jcarousel(s.target>0?"hasNext":"hasPrev");else{var r="object"!=typeof s.target?e.jcarousel("items").eq(s.target):s.target;i=e.jcarousel("target").index(r)>=0}return this._active!==i&&(this._trigger(i?"active":"inactive"),this._active=i),this}})}(jQuery),function(t){"use strict";t.jCarousel.plugin("jcarouselPagination",{_options:{perPage:null,item:function(t){return'<a href="#'+t+'">'+t+"</a>"},event:"click",method:"scroll"},_pages:{},_items:{},_currentPage:null,_init:function(){this.onDestroy=t.proxy(function(){this._destroy(),this.carousel().one("jcarousel:createend",t.proxy(this._create,this))},this),this.onReload=t.proxy(this._reload,this),this.onScroll=t.proxy(this._update,this)},_create:function(){this.carousel().one("jcarousel:destroy",this.onDestroy).on("jcarousel:reloadend",this.onReload).on("jcarousel:scrollend",this.onScroll),this._reload()},_destroy:function(){this._clear(),this.carousel().off("jcarousel:destroy",this.onDestroy).off("jcarousel:reloadend",this.onReload).off("jcarousel:scrollend",this.onScroll)},_reload:function(){var i=this.options("perPage");if(this._pages={},this._items={},t.isFunction(i)&&(i=i.call(this)),null==i)this._pages=this._calculatePages();else for(var s,e=parseInt(i,10)||0,r=this.carousel().jcarousel("items"),n=1,o=0;;){if(s=r.eq(o++),0===s.length)break;this._pages[n]=this._pages[n]?this._pages[n].add(s):s,0===o%e&&n++}this._clear();var l=this,a=this.carousel().data("jcarousel"),h=this._element,u=this.options("item");t.each(this._pages,function(i,s){var e=l._items[i]=t(u.call(l,i,s));e.on(l.options("event")+".jcarouselpagination",t.proxy(function(){var t=s.eq(0);if(a.circular){var e=a.index(a.target()),r=a.index(t);parseFloat(i)>parseFloat(l._currentPage)?e>r&&(t="+="+(a.items().length-e+r)):r>e&&(t="-="+(e+(a.items().length-r)))}a[this.options("method")](t)},l)),h.append(e)}),this._update()},_update:function(){var i,s=this.carousel().jcarousel("target");t.each(this._pages,function(t,e){return e.each(function(){return s.is(this)?(i=t,!1):void 0}),i?!1:void 0}),this._currentPage!==i&&(this._trigger("inactive",this._items[this._currentPage]),this._trigger("active",this._items[i])),this._currentPage=i},items:function(){return this._items},_clear:function(){this._element.empty(),this._currentPage=null},_calculatePages:function(){for(var t,i=this.carousel().data("jcarousel"),s=i.items(),e=i.clipping(),r=0,n=0,o=1,l={};;){if(t=s.eq(n++),0===t.length)break;l[o]=l[o]?l[o].add(t):t,r+=i.dimension(t),r>=e&&(o++,r=0)}return l}})}(jQuery),function(t){"use strict";t.jCarousel.plugin("jcarouselAutoscroll",{_options:{target:"+=1",interval:3e3,autostart:!0},_timer:null,_init:function(){this.onDestroy=t.proxy(function(){this._destroy(),this.carousel().one("jcarousel:createend",t.proxy(this._create,this))},this),this.onAnimateEnd=t.proxy(this.start,this)},_create:function(){this.carousel().one("jcarousel:destroy",this.onDestroy),this.options("autostart")&&this.start()},_destroy:function(){this.stop(),this.carousel().off("jcarousel:destroy",this.onDestroy)},start:function(){return this.stop(),this.carousel().one("jcarousel:animateend",this.onAnimateEnd),this._timer=setTimeout(t.proxy(function(){this.carousel().jcarousel("scroll",this.options("target"))},this),this.options("interval")),this},stop:function(){return this._timer&&(this._timer=clearTimeout(this._timer)),this.carousel().off("jcarousel:animateend",this.onAnimateEnd),this}})}(jQuery);/**
* jquery.matchHeight-min.js master
* http://brm.io/jquery-match-height/
* License: MIT
*/
(function(c){var n=-1,f=-1,g=function(a){return parseFloat(a)||0},r=function(a){var b=null,d=[];c(a).each(function(){var a=c(this),k=a.offset().top-g(a.css("margin-top")),l=0<d.length?d[d.length-1]:null;null===l?d.push(a):1>=Math.floor(Math.abs(b-k))?d[d.length-1]=l.add(a):d.push(a);b=k});return d},p=function(a){var b={byRow:!0,property:"height",target:null,remove:!1};if("object"===typeof a)return c.extend(b,a);"boolean"===typeof a?b.byRow=a:"remove"===a&&(b.remove=!0);return b},b=c.fn.matchHeight=
function(a){a=p(a);if(a.remove){var e=this;this.css(a.property,"");c.each(b._groups,function(a,b){b.elements=b.elements.not(e)});return this}if(1>=this.length&&!a.target)return this;b._groups.push({elements:this,options:a});b._apply(this,a);return this};b._groups=[];b._throttle=80;b._maintainScroll=!1;b._beforeUpdate=null;b._afterUpdate=null;b._apply=function(a,e){var d=p(e),h=c(a),k=[h],l=c(window).scrollTop(),f=c("html").outerHeight(!0),m=h.parents().filter(":hidden");m.each(function(){var a=c(this);
a.data("style-cache",a.attr("style"))});m.css("display","block");d.byRow&&!d.target&&(h.each(function(){var a=c(this),b=a.css("display");"inline-block"!==b&&"inline-flex"!==b&&(b="block");a.data("style-cache",a.attr("style"));a.css({display:b,"padding-top":"0","padding-bottom":"0","margin-top":"0","margin-bottom":"0","border-top-width":"0","border-bottom-width":"0",height:"100px"})}),k=r(h),h.each(function(){var a=c(this);a.attr("style",a.data("style-cache")||"")}));c.each(k,function(a,b){var e=c(b),
f=0;if(d.target)f=d.target.outerHeight(!1);else{if(d.byRow&&1>=e.length){e.css(d.property,"");return}e.each(function(){var a=c(this),b=a.css("display");"inline-block"!==b&&"inline-flex"!==b&&(b="block");b={display:b};b[d.property]="";a.css(b);a.outerHeight(!1)>f&&(f=a.outerHeight(!1));a.css("display","")})}e.each(function(){var a=c(this),b=0;d.target&&a.is(d.target)||("border-box"!==a.css("box-sizing")&&(b+=g(a.css("border-top-width"))+g(a.css("border-bottom-width")),b+=g(a.css("padding-top"))+g(a.css("padding-bottom"))),
a.css(d.property,f-b+"px"))})});m.each(function(){var a=c(this);a.attr("style",a.data("style-cache")||null)});b._maintainScroll&&c(window).scrollTop(l/f*c("html").outerHeight(!0));return this};b._applyDataApi=function(){var a={};c("[data-match-height], [data-mh]").each(function(){var b=c(this),d=b.attr("data-mh")||b.attr("data-match-height");a[d]=d in a?a[d].add(b):b});c.each(a,function(){this.matchHeight(!0)})};var q=function(a){b._beforeUpdate&&b._beforeUpdate(a,b._groups);c.each(b._groups,function(){b._apply(this.elements,
this.options)});b._afterUpdate&&b._afterUpdate(a,b._groups)};b._update=function(a,e){if(e&&"resize"===e.type){var d=c(window).width();if(d===n)return;n=d}a?-1===f&&(f=setTimeout(function(){q(e);f=-1},b._throttle)):q(e)};c(b._applyDataApi);c(window).bind("load",function(a){b._update(!1,a)});c(window).bind("resize orientationchange",function(a){b._update(!0,a)})})(jQuery);	jQuery.fn.center = function () {
		this.css("position","absolute");
		this.css("top", Math.max(0, (($(window).height() - $(this).outerHeight()) / 2) + $(window).scrollTop()) + "px");
		this.css("left", Math.max(0, (($(window).width() - $(this).outerWidth()) / 2) + $(window).scrollLeft()) + "px");
		return this;
	};

	function SpringerOverlay (layerElement, layerBgElement, targetOpacity) {
		this.layerElement = layerElement;
		this.layerBgElement = layerBgElement;
		this.closeTrigger = $(layerElement).find(".closeTrigger:first");
		this.targetOpacity = targetOpacity;

		this.closeLayerAndBackground = function () {
			var layerElement = this.layerElement;
			layerElement.hide();
			layerBgElement.finish().fadeOut("fast", function() {
				if (layerElement.is(":visible")) {
					layerElement.hide(); // Make sure that the layerElement is hidden when the background is gone
				}
			});
		};

		this.showLayerAndBackground = function () {
			this.layerBgElement.finish().fadeTo("fast", this.targetOpacity, function() {
				layerElement.center();
				layerElement.show();
			});
		};

		SpringerOverlay.prototype.bindEvents = function () {
			var springerOverlay = this;
			$(window).on("resize", function() {
				var layer = springerOverlay.layerElement;
				if (layer.is(":visible")) {
					layer.center();
				}
			});
			this.layerBgElement.on("click", function () {
				springerOverlay.closeLayerAndBackground();
			});
			this.closeTrigger.on("click", function() {
				springerOverlay.closeLayerAndBackground();
				return false;
			});
			$(document).keyup(function(e) {
				if (e.keyCode == 27)/* ESC */ {
					springerOverlay.closeLayerAndBackground();
				}
			});
		};

		SpringerOverlay.prototype.show = function () {
			this.showLayerAndBackground();
		};
		SpringerOverlay.prototype.hide = function () {
			this.closeLayerAndBackground();
		};

		SpringerOverlay.prototype.finishFading = function () {
			this.layerBgElement.finish();
			this.layerElement.finish();
		};
	}/**
 * adds 'wicket' header params as url params , since some proxies steal the header attrib  
 */
$(function($) {
	try {
		var orgFunc = Wicket.Ajax.Call.prototype.doAjax;
		var orgBaseURL = Wicket.Ajax.baseUrl;
		Wicket.Ajax.Call.prototype.doAjax = function(attrs){
			if(attrs.u.indexOf('?') >= 0) {
				attrs.u = attrs.u + '&'
			} else {
				attrs.u = attrs.u + '?'
			}
			attrs.u = attrs.u + 'wicket-ajax=true' + '&wicket-ajax-baseurl=' + encodeURIComponent(orgBaseURL);
			orgFunc.call(this, attrs);
		};
	} catch (e) {}
});

$(function(){
	/* Help tooltips
	------------------------------------------------------------------------------*/
	$('body').delegate('.help', 'mouseover mouseout', function(e) {
		if (e.type === 'mouseover'){
			var tooltip = $('.tooltip',this);
			var linkPos = $(this).offset();
			tooltip.clone().appendTo('body').css({
				left : linkPos.left + 15,
				top : linkPos.top + 30
			}).show().addClass('tooltip-clone');
		} else {
			$('.tooltip-clone').remove();
		}
	});
	/* Help tooltips
	 ------------------------------------------------------------------------------*/
	$('body').delegate('.btn-help', 'mouseover mouseout', function(e) {
		if (e.type === 'mouseover') {
			var tooltip = $('.tooltip', this);
			var linkPos = $(this).offset();
			tooltip.clone().appendTo('body').css({
				left : linkPos.left + 15,
				top : linkPos.top + 34
			}).show().addClass('tooltip-clone');
		} else {
			$('.tooltip-clone').remove();
		}
	});

    /* Autosuggest
    ------------------------------------------------------------------------------*/
    searchWidth = $('.search-send_field').width() - 53;
    searchWidth = $('.search-send_field').width() - 53;
    $(window).resize(function(){
        $('.ui-autocomplete').hide();
        searchWidth = $('.search-send_field').width() - 53;
    });
    $('.ui-autocomplete').width(searchWidth);

	/* LAZY LOAD IMAGE */
	lazyLoadImages();
	
	/* PREVIEW STUDIO MENU */
	if(!inIframe() && $.contextMenu) {
		var studioWindow = null;
		$.contextMenu({
	        selector: '.cms-studio-maintainable', 
	        className: 'context-menu-title-studio',
	        build: function($trigger, e) {
	            // this callback is executed every time the menu is to be shown
	            // its results are destroyed every time the menu is hidden
	            // e is the original contextmenu event, containing e.pageX and e.pageY (amongst other data)
	        	var ret = {
	                callback: function(key, options) {
	                	if(studioWindow != null && !studioWindow.closed) {
	                		// the remote control only works in a fresh window
	                		studioWindow.close();
	                	}
	                	studioWindow = window.open(options.items[key].url, 'studioWindow');
	                	window.setTimeout(function() {
	                		// close window that triggers the studio
		                	if(studioWindow != null && !studioWindow.closed) {
		                		studioWindow.close();
		                	}
	                	}, 4000);
	                },
	                items: {}
	            };
	        	var id = $trigger.attr('id');
	    		ret.items[id] = cmStudioLinks[id];
	    		var parents = $trigger.parents('.cms-studio-maintainable').each(function(){
	    			id = $(this).attr('id');
	    			ret.items[id] = cmStudioLinks[id];
	    		});
	        	return ret;
	        }
	    });
	}
	
});


function alignCampaignIntroduction() {
    $('.cms-campaign-placeholder').each(function() {
    	$(this).height($(this).prev().height());
    });
}

function adaptUI(ctx /* the DOM context */){
	
	/* Expanders
	------------------------------------------------------------------------------*/
	$('.expander', ctx).not('.expander-empty').each(function() {
	    //set up
	    //$(this).not('.expander-open').find('.expander-content', this).hide();
		// interaction
		$('.expander-title', this).css('cursor', 'pointer').click(function() {
			var $expander = $(this).closest('.expander');
			if ($('.expander-content', $expander).is(':visible')) {
				$expander.find('.expander-content').slideUp('normal', function(){
				    $expander.removeClass('expander-open');
				});
			} else {
				$expander.find('.expander-content').slideDown('normal', function(){
				    $expander.addClass('expander-open');
			    });
			}
		});
	});
	
	/* Expanders (CMS)
	------------------------------------------------------------------------------*/
	$('ul > .flapContent--paragraph, ol > .flapContent--paragraph', ctx).parent().addClass('flapContent');
	$('table.flapContent--paragraph', ctx).wrap('<div class="flapContent"></div>'); // table must be wrapped in div to achieve padding
	
	$('.flapHead', ctx).click(function() {
		$(this).toggleClass('showContent');
		var flapcontent = $(this).next();
		// iterate through all sequential flapContent and stop on end or next flapHead
		while(flapcontent.length > 0 && !$(flapcontent).hasClass('flapHead') && $(flapcontent).hasClass('flapContent')) {
			$(flapcontent).slideToggle();
			flapcontent = $(flapcontent).next();
		}
	});
	
	/* mark every first flap head on the entire page */
	$('.flapHead', ctx).first().addClass('firstFlapHead');
	$(':first-child.flapHead', ctx).addClass('firstFlapHead');
	$('.flapHead', ctx).prev(":not(.flapContent)").next().addClass('firstFlapHead');
	
	/* mark every last flap content on the entire page */
	$('.flapContent', ctx).next(':not(.flapContent)').prev().addClass('lastFlapContent');
	$(':last-child.flapContent', ctx).addClass('lastFlapContent');
	$('.flapContent', ctx).last().addClass('lastFlapContent');

	/* insert a seperator between the flap items */
	$('.firstFlapHead', ctx).before('<div class="flapSeparator"></div>');
	$('.lastFlapContent', ctx).after('<div class="flapSeparator"></div>');


    /* Expander close on mobile
    ------------------------------------------------------------------------------*/
	if($(window).width() < 600) {
		$('.expander', ctx).removeClass('expander-open');
	}

	/* ARTNOLOGY CUSTOM */
	/* mark external links
	--------------------------------------------------------------------------------*/
	var currentHost = window.location.protocol + '//' + window.location.host;
	$('.cms-article a, .cms-person a, .cms-toc a, .link-text a, .cms-linklists a, .cms-widget-footer a, .cms-myspringer-login a', ctx)
		.each(function(){
			if($(this).prop('href').indexOf(currentHost) == 0) {
				$(this).addClass('internal');
			} else {
				$(this).addClass('external');
			}
		})
		.filter('[target]')
		.each(function() {
			$(this).removeAttr('target');
		});
	
	// whitelist some external links and show them as internal
	$('a.external[href*="mailto:"], a.external[href*="www.springer.com"], a.external[href*="/springer-cms/rest"], a.external[href*="palgrave.com"], a.external[href*="apress.com"], a.external[href*="springernature.com"], .hp-bigtext a.external', ctx).removeClass('external').addClass('internal');
	/* /mark external links */
	
	/* Add button styling to links in Articles 
	----------------------------------------------------------------------------------- */
	$('.cms-richtext .inline-content', ctx).each(function() {
		$(this).addClass('btn btn-primary btn-monster');
	});
	
	/* list with only links should have styling without bullets but links with chevrons
	----------------------------------------------------------------------------------- */
	$('.cms-article:not(.cms-large-lists) ul:has(a), .cms-person ul:has(a), .cms-myspringer-login ul:has(a)', ctx).each(function() {
		if($(this).children().length == $(this).children(':has(a)').length) {
			var applyLinklist = true;
			$(this).children().each(function() {
				if($(this).html().trim().indexOf('<a') != 0) {
					applyLinklist = false;
				}
			});
			if(applyLinklist) {
				$(this).addClass('cms-link-list');
			}
		}
	});
	
	/* Teasers clickable all over
	--------------------------------------------------------------------------------*/
	$('.cms-teasable-content, .cms-teaser, .cms-teaser-sidebar', ctx).children('.link-text:has(a[href])').each(function() {
		$(this).siblings(':not(a)').wrapInner( $('<a></a>', { "href": $('a',this).attr('href'), "class": "defensive-link" }) );
	});
	
	/* Add missing separator
	----------------------------------------------*/
	$('.twoColumnRight--paragraph + .twoColumnLeft--paragraph').before('<p class="twoColumnSeparatorLeft--paragraph--noborder"></p>');
	
	/* SMOOTH SCROLLING
	--------------------------------------------------------------------------------*/
	$('a[href^=#]:not([href=#])', ctx).click(function() {
	    if (window.location.pathname.replace(/^\//,'') == this.pathname.replace(/^\//,'') && window.location.hostname == this.hostname) {
	      var target = $(this.hash);
	      var hash = this.hash;
	      target = target.length ? target : $('[name=' + this.hash.slice(1) +']');
	      if (target.length) {
	    	  if($(this).closest('.sba_pagination').length == 0) {
		        $('html,body').animate({scrollTop: target.offset().top }, 500, function() { window.location = hash; });
		        return false;
	    	  }
	      }
	    }
	  });

	/* activate enhanced select-box, http://harvesthq.github.com/chosen/) */
	$(".country_select, .state_select, .sap_title_select, .school_select", ctx).chosen({
		no_results_text: "No results matched", width: "98.5%"
	});
	
	$(".salutation_select, .shipping-select", ctx).chosen({
		no_results_text: "No results matched", width: "98.5%", disable_search_threshold: 10
	});
	
	/* auto start automatic carousels */
	initializePagingCarousel('.cms-carousel-automatic');
	
	$(window).resize(alignCampaignIntroduction);
	alignCampaignIntroduction();
	
	
	$(document).on('mouseover focus', '.grid-tiles .tile-toggle', function() {
		if (!isTouchDevice) {
			var that = $(this);
			that.addClass('toggled').removeClass('click-prevent');
		}
	});
	$(document).on('mouseout blur', '.grid-tiles .tile-toggle', function() {
		if (!isTouchDevice) {
			var that = $(this);
			that.removeClass('toggled').addClass('click-prevent');
		}
	});
	
	/* tiles */
	$('.cms-hp-tile .tile-toggle').hover(
			function() {
				$(this).addClass('toggled');
			},
			function() {
				$(this).removeClass('toggled');
			}
	);
}

inIframe = function() {
	try {
	    return window.self !== window.top;
	} catch (e) {
	    return true;
	}
}


blockUi = function(selector) {
	$(selector).block({ 
		message: '<img src="/spcom/images/loading.gif" alt="loading..." />', 
		css: { width: 'auto', padding: '2em 15%', border: 'none', 'border-radius': '3px'},
		overlayCSS: {opacity: '0.1', 'box-shadow': '3px 3px 5px #000', 'border-radius': '3px'} 
	});
};

unblockUi = function(selector) {
	$(selector).unblock();
};

/* Menu on dropdown buttons
------------------------------------------------------------------------------*/
dropdownButtons = function() {
    btnDd = $('.btn-dd:not(.btn-dd-attached)');
    btnDd.click(function(event) {
        $(this).toggleClass('act')
               .find('.menu')
               .slideToggle('fast');
    	}).addClass('btn-dd-attached');
};
closeDropdownButtons = function() {
    $('.btn-dd').removeClass('act').find('.menu').hide();
};

function lazyLoadImages() {
	$('img.lazy').not('.lazy-marked').show().lazyload().addClass('lazy-marked');
}

function shopLandingPageActions() {
	dropdownButtons();
	lazyLoadImages();
}

/* NOTE: keep in sync with frontend main.js */
function writeUserTokenCookie(cookiename, token, domain) {
	$.cookie.raw = true;
	var cookieValue = $.cookie(cookiename);
	
	if (cookieValue){
		$.removeCookie(cookiename,{ domain : domain, path: '/' });
	}
	
	$.cookie(cookiename, token,{ domain : domain, path: '/' });
}

/* VIDEO CAROUSEL */
function configureVideoCarousel() {
	var element = $(this);
	var width = element.width();
	var elemWidth = 0;
	var elemPadding = 0;
	if(width > 350) {
		elemPadding = width * .02;
		elemWidth = width / 3 - elemPadding * 2;
	} else {
		elemPadding = width * .03;
		elemWidth = width / 2 - elemPadding * 2;
	}
	element.jcarousel('items').css('width', Math.floor(elemWidth) + 'px').css('padding', '0 ' + Math.floor(elemPadding) + 'px');
}

function initializeVideoCarousel(containerId) {
	$('#'+containerId + ' .video-carousel')
		.on('jcarousel:create jcarousel:reload', configureVideoCarousel)
		.jcarousel({wrap: 'both'});
	$('#'+containerId + ' .video-carousel-container .control-prev').click(function(e) {
		e.preventDefault();
	    $('#'+containerId + ' .video-carousel').jcarousel('scroll', '-=1');
	});
	$('#'+containerId + ' .video-carousel-container .control-next').click(function(e) {
		e.preventDefault();
	    $('#'+containerId + ' .video-carousel').jcarousel('scroll', '+=1');
	});
	$('#'+containerId + ' .video-carousel li').click(function(e) {
		var id = 'video-'+containerId;
		var url = $(this).attr('data-href') + '&jsdiv=' + id;

		// set height to current height to prevent jumping of the page on temporary height=0 when replacing the player
		var oldHeight = $('#'+containerId + ' .video-wrapper').height();
		if(oldHeight > 20) {
			$('#'+containerId + ' .video-wrapper').css('height', oldHeight + 'px');
		}
	    $('#'+containerId + ' .video-wrapper').html('<div id="'+id+'"></div><script type="text/javascript" src="' + url + '"></script>');
	    $('#'+containerId + ' .video-title').html($(this).children('h4').html());
	    var videoDescription = $(this).children('div.description').html();
	    if(typeof videoDescription === 'undefined') {
	    	videoDescription = '';
	    }
	    $('#'+containerId + ' .video-description').html(videoDescription);
	    window.setTimeout(function(){$('#'+containerId + ' .video-wrapper').css('height', 'auto')}, 1000);
	});
	$('#'+containerId + ' .video-carousel li:first-child').click();
}
/*/VIDEO CAROUSEL */

/* PAGING CAROUSEL */
function configurePagingCarousel() {
	var $element = $(this);

	$element.css('width', 'auto');
	var width = $element.width();
	var minItemWidth = 200;
	if(typeof $element.data('item-min-width') !== 'undefined') {
		minItemWidth = $element.data('item-min-width');
	}

	if($element.parents('.cms-client-palgrave').length || $element.parents('.cms-client-apress').length) {
		configureCarouselConfigForBorderBox($element, width, minItemWidth);
	} else {
		configureCarouselConfigForContentBox($element, width, minItemWidth);
	}
}

function configureCarouselConfigForBorderBox($element, width, minItemWidth) {
	var i;
	var config = [];
	for(i = 20; i > 0; i--) {
		if(i === 1) {
			config.push({width : 0, count : i});
		} else {
			config.push({width : i * minItemWidth, count : i});
		}
	}
	var elemWidth = 0;
	var elemPadding = 0;
	var i;
	for (i = 0; i < config.length; i++) {
		if (width >= config[i].width) {
			elemWidth = Math.floor(width / config[i].count);
			break;
		}
	}
	$element.data('carouselStep', config[i].count);
	$element.css('width', ((elemWidth) * config[i].count) + 'px');
	$element.jcarousel('items').css('width', elemWidth + 'px');
}

function configureCarouselConfigForContentBox($element, width, minItemWidth) {
	var i;
	var config = [];
	for(i = 20; i > 0; i--) {
		if(i === 1) {
			config.push({width : 0, count : i, padding : 0.03});
		} else {
			config.push({width : i * minItemWidth, count : i, padding : 0.03});
		}
	}
	var elemWidth = 0;
	var elemPadding = 0;
	var i;
	for (i = 0; i < config.length; i++) {
		if (width >= config[i].width) {
			elemPadding = Math.floor(width * config[i].padding);
			elemWidth = Math.floor(width / config[i].count - elemPadding);
			break;
		}
	}
	$element.data('carouselStep', config[i].count);
	$element.css('width', ((elemWidth + elemPadding) * config[i].count) + 'px');
	$element.jcarousel('items').css('width', elemWidth + 'px').css('padding', '0 ' + elemPadding + 'px 0 0');
}

function initializePagingCarousel(containerSelector) {
	$(containerSelector).each(function() {
		var $container = $(this).removeClass('loading-indicator');
		var $carousel = $('.carousel', $container);
		var $counter = $('.carousel-counter', $container);
		if($('ul li', $carousel).length > 0) {
			$container.show(); // in case it was hidden
			var navigateCarousel = function(e, plusOneOrMinusOne) {
				e.preventDefault();
				var step = $carousel.data('carouselStep');
				$carousel.jcarousel('scroll', (plusOneOrMinusOne > 0 ? '+' : '-') + '=' + $carousel.data('carouselStep'));
				$carousel.data('scrollDirection', plusOneOrMinusOne);
			};
			$carousel
				.on('jcarousel:create jcarousel:reload', configurePagingCarousel)
				.jcarousel({
					wrap: 'both',
					animation: {
						duration: 400,
						easing: 'linear',
						complete: function() {
							updateCarouselCounter($counter, $carousel, $carousel.data('scrollDirection') * $carousel.data('carouselStep'));
						}
					}
				});
			$('.carousel-container .control-prev', $container).click(function(e) {
				navigateCarousel(e, -1);
			});
			$('.carousel-container .control-next', $container).click(function(e) {
				navigateCarousel(e, 1);
			});
			updateCarouselCounter($counter, $carousel, 0);
		}
	});
}

function updateCarouselCounter($counter, $carousel, change) {
	$('img.lazy', $carousel).trigger('appear');
	if($counter.length == 1) {
		if(typeof $counter.data('total') === 'undefined') {
			$counter.data('total', $('li', $carousel).length);
			$('.total', $counter).html($counter.data('total'));
		}
		if(typeof $counter.data('current') === 'undefined') {
			$counter.data('current', 1);
		}
	
		if(change != 0) {
			var current = $counter.data('current');
			var total = $counter.data('total');
			var newCurrent = current;
			
			var lastSlideNext = (current + change > total);
			var lastSlide = (current + change === total);
			var firstSlidePrev = (current === 1 && change < 0);
			var firstSlide = (current + change <= 1);
			
			// the following is only characteristic to wrap: 'both'
			// set to one whenever a boundary is crossed
			if(lastSlideNext) {
				newCurrent = 1;
			} else if(lastSlide) {
				newCurrent = total;
			} else if(firstSlidePrev) {
				// at min limit > scroll to last
				newCurrent = 1 + total + change;
			} else if(firstSlide) {
				// get first scroll > gto min 
				newCurrent = 1;
			} else {
				// in the middle > just add the change
				newCurrent = current + change;
			}
			$counter.data('current', newCurrent);
		}
		
		$('.current', $counter).html($counter.data('current'));
	}
}
/*/PAGING CAROUSEL */

function navigationExpander() {	
	$('.tree-nav').each(
		function() {
			var showNav = $(this).find('.show-nav');
			var collapseNav = $(this).find('.collapse-nav');
			var hideNav =  $(this).find('.hide-nav');
			$(showNav).click(
			    function() {
			        $(hideNav).slideDown('slow', function() {
				        $(collapseNav).css('display','block');
				    	$(showNav).hide();
			        });
			        return false;
			    }
			);
			$(collapseNav).click(
			    function() {
			        $(hideNav).slideUp('slow', function() {
				        $(showNav).css('display','block');
				        $(collapseNav).hide();
			        });
			        return false;
			    }
			);
		}
	);
}

$(document).ready(function(){
	adaptUI();
	shopLandingPageActions();
	navigationExpander();
});



/**
 ************ Don't change anything beyond this line ************
 ********************* Start webtrekk_v4.js *********************
 */var webtrekkUnloadObjects=webtrekkUnloadObjects||[],webtrekkLinktrackObjects=webtrekkLinktrackObjects||[],webtrekkHeatmapObjects=webtrekkHeatmapObjects||[],webtrekkV3=function(g){var u=function(a,b){"1"!=f.cookie||(f.optOut||f.deactivatePixel)||f.firstParty();var c=b?b:f.formObject&&"noForm"!=a?"form":"link";!1!=f.beforeUnloadPixel?f.beforeUnloadPixel():"form"==c&&f.executePlugin(f.getPluginConfig("form","before"));var d="";if(f.config.linkId&&(d+="&ct="+f.wtEscape(f.maxlen(f.wtUnescape(f.config.linkId),
255)))){f.linktrackOut&&(d+="&ctx=1");var e=f.ccParams;"string"==typeof e&&""!=e&&(d+=e)}if(f.wtEp)if(f.wtEpEncoded)d+=f.wtEp;else if(e=f.wtEp,"string"==typeof e&&""!=e)for(var e=e.split(/;/),t=0;t<e.length;t++)if(f.wtTypeof(e[t])){var k=e[t].split(/=/);f.checkSC("custom")&&(k[1]=f.decrypt(k[1]));k[1]=f.wtEscape(k[1]);d+="&"+k[0]+"="+k[1]}"noForm"!=a&&(d+=f.checkFormTracking());""!=d&&(f.quicksend(f.wtEscape(f.contentId.split(";")[0])+",1,"+f.baseparams(),d),f.config.linkId="",f.ccParams="",f.wtEp=
"");!1!=f.afterUnloadPixel?f.afterUnloadPixel():"form"==c&&f.executePlugin(f.getPluginConfig("form","after"))},q=function(a){var b,c,d=document.getElementById(f.heatmapRefpoint);c=d&&null!==d?b=0:b=-1;if(d&&null!==d&&f.wtTypeof(d.offsetLeft))for(;d;)b+=0<=d.offsetLeft?d.offsetLeft:0,c+=0<=d.offsetTop?d.offsetTop:0,d=d.offsetParent;var e=d=0;a||(a=window.event);if(a.pageX||a.pageY)d=a.pageX,e=a.pageY;else if(a.clientX||a.clientY)if(d=a.clientX,e=a.clientY,f.isIE)if(0<document.body.scrollLeft||0<document.body.scrollTop)d+=
document.body.scrollLeft,e+=document.body.scrollTop;else if(0<document.documentElement.scrollLeft||0<document.documentElement.scrollTop)d+=document.documentElement.scrollLeft,e+=document.documentElement.scrollTop;a=0;a=f.isIE?document.body.clientWidth:self.innerWidth-16;var t=!0;if(d>=a||!f.sentFullPixel)t=!1;(0<=c||0<=b)&&(d>b&&e>c)&&(d="-"+(d-b),e="-"+(e-c));t&&"1"==f.heatmap&&(f.executePlugin(f.getPluginConfig("heatmap","before")),f.quicksend(f.wtEscape(f.contentId.split(";")[0])+","+d+","+e,"",
"hm"+(f.fileSuffix?".pl":"")),f.executePlugin(f.getPluginConfig("heatmap","after")))},r=function(){"undefined"!==typeof wt_heatmap?window.setTimeout(function(){wt_heatmap()},1E3):("undefined"===typeof wt_heatmap_retry&&(window.wt_heatmap_retry=0),wt_heatmap_retry++,60>wt_heatmap_retry&&window.setTimeout(function(){r()},1E3))},l=function(){"undefined"!==typeof wt_overlay?window.setTimeout(function(){wt_overlay()},1E3):("undefined"===typeof wt_overlay_retry&&(window.wt_overlay_retry=0),wt_overlay_retry++,
60>wt_overlay_retry&&window.setTimeout(function(){l()},1E3))},s=function(a,b){var c=f.urlParam(location.href,a,!1),d=f.urlParam(location.href,"wt_t",!1),e=(new Date).getTime(),t=RegExp(b),d=d?parseInt(d)+9E5:e-9E5;return c&&t.test(c)&&d>e?c:!1},n=function(a){if(a&&"2"==a.substring(0,1)){a=parseInt(a.substring(1,11)+"000");a=new Date(a);var b=a.getFullYear()+"",b=b+(9>a.getMonth()?"0":""),b=b+(a.getMonth()+1),b=b+(9>a.getDate()?"0":""),b=b+a.getDate(),b=b+(9>a.getHours()?"0":""),b=b+a.getHours(),b=
b+(9>a.getMinutes()?"0":"");return b+=a.getMinutes()}return""},h=webtrekkConfig,f=this;g||(g=h);this.defaultAttribute="contentId linkId trackId trackDomain domain linkTrack linkTrackParams linkTrackPattern linkTrackReplace linkTrackDownloads linkTrackIgnorePattern customParameter crmCategory urmCategory customClickParameter customSessionParameter customTimeParameter customCampaignParameter customEcommerceParameter orderValue currency orderId product productCost productQuantity productCategory productStatus couponValue customerId contentGroup mediaCode mediaCodeValue mediaCodeCookie campaignId campaignAction internalSearch customSid customEid cookieDomain cookieEidTimeout cookieSidTimeout forceNewSession xwtip xwtua xwtrq xwteid xwtstt mediaCodeFrames framesetReferrer forceHTTPS secureConfig heatmap pixelSampling form formFullContent formAnonymous disableOverlayView beforeSendinfoPixel afterSendinfoPixel beforeUnloadPixel afterUnloadPixel xlc xlct xlcv ignorePrerendering isIE isOpera isSafari isChrome isFirefox email emailRID emailOptin firstName lastName telefon gender birthday birthdayJ birthdayM birthdayD country city postalCode street streetNumber validation fileSuffix".split(" ");
this.cookie=g.cookie?g.cookie:h.cookie?h.cookie:"3";this.optoutName=g.optoutName?g.optoutName:h.optoutName?h.optoutName:"webtrekkOptOut";this.paramFirst=g.paramFirst?g.paramFirst:h.paramFirst?h.paramFirst:"";this.maxRequestLength=g.maxRequestLength?g.maxRequestLength:h.maxRequestLength?h.maxRequestLength:7168;this.plugins=g.plugins&&""!=g.plugins?g.plugins:h.plugins&&""!=h.plugins?h.plugins:"Adobe Acrobat;Windows Media Player;Shockwave Flash;RealPlayer;QuickTime;Java;Silverlight".split(";");"string"==
typeof this.plugins&&(this.plugins=this.plugins.split(";"));this.heatmapRefpoint=g.heatmapRefpoint?g.heatmapRefpoint:h.heatmapRefpoint?h.heatmapRefpoint:"wt_refpoint";this.linkTrackAttribute=g.linkTrackAttribute?g.linkTrackAttribute:h.linkTrackAttribute?h.linkTrackAttribute:"name";this.delayLinkTrack=g.delayLinkTrack?g.delayLinkTrack:h.delayLinkTrack?h.delayLinkTrack:!1;this.delayLinkTrackTime=g.delayLinkTrackTime?g.delayLinkTrackTime:h.delayLinkTrackTime?h.delayLinkTrackTime:200;this.noDelayLinkTrackAttribute=
g.noDelayLinkTrackAttribute?g.noDelayLinkTrackAttribute:h.noDelayLinkTrackAttribute?h.noDelayLinkTrackAttribute:!1;this.formAttribute=g.formAttribute?g.formAttribute:h.formAttribute?h.formAttribute:"name";this.formFieldAttribute=g.formFieldAttribute?g.formFieldAttribute:h.formFieldAttribute?h.formFieldAttribute:"name";this.formValueAttribute=g.formValueAttribute?g.formValueAttribute:h.formValueAttribute?h.formValueAttribute:"value";this.formFieldDefaultValue=g.formFieldDefaultValue?g.formFieldDefaultValue:
h.formFieldDefaultValue?h.formFieldDefaultValue:{};this.formPathAnalysis=g.formPathAnalysis?g.formPathAnalysis:h.formPathAnalysis?h.formPathAnalysis:!1;this.reporturl=g.reporturl?g.reporturl:h.reporturl?h.reporturl:"report2.webtrekk.de/cgi-bin/wt";this.updateCookie=g.updateCookie?g.updateCookie:h.updateCookie?h.updateCookie:!0;this.executePluginFunction=g.executePluginFunction?g.executePluginFunction:h.executePluginFunction?h.executePluginFunction:"";this.linktrackOut=this.cookieOne=this.sampleCookieString=
this.lastVisitContact=this.firstVisitContact=this.eid=this.optOut=this.deactivateRequest=this.deactivatePixel=!1;this.linktrackNamedlinksOnly=!0;this.sentFullPixel=this.ccParams=!1;this.sentCampaignIds={};this.config=this.browserLang=this.formSubmit=this.formFocus=this.formName=this.formObject=this.gatherFormsP=this.overlayOn=this.heatmapOn=this.trackingSwitchMediaCodeTimestamp=this.trackingSwitchMediaCodeValue=this.trackingSwitchMediaCode=this.wtEpEncoded=this.wtEp=!1;this.unloadInstance=webtrekkUnloadObjects.length;
this.plugin={};this.heatmapCounter=this.formCounter=this.linkCounter=this.clickCounter=this.pageCounter=0;this.browserLang=!1;"string"==typeof navigator.language?this.browserLang=navigator.language.substring(0,2):"string"==typeof navigator.userLanguage&&(this.browserLang=navigator.userLanguage.substring(0,2));this.jsonPara={ck:["customClickParameter",{}],cp:["customParameter",{}],cs:["customSessionParameter",{}],ce:["customTimeParameter",{}],cb:["customEcommerceParameter",{}],vc:["crmCategory",{}],
uc:["urmCategory",{}],ca:["productCategory",{}],cc:["customCampaignParameter",{}],cg:["contentGroup",{}],ct:["linkId",""],ov:["orderValue",""],cr:["currency",""],oi:["orderId",""],ba:["product",""],co:["productCost",""],qn:["productQuantity",""],st:["productStatus",""],cd:["customerId",""],is:["internalSearch",""],mc:["campaignId",""],mca:["campaignAction",""]};this.generateDefaultConfig=function(a,b){for(var c=0;c<this.defaultAttribute.length;c++){var d=this.defaultAttribute[c];this[d]=a[d]?a[d]:
b[d]?b[d]:!1}};this.generateDefaultConfig(g,h);this.campaignAction=g.campaignAction?g.campaignAction:h.campaignAction?h.campaignAction:"click";"undefined"===typeof this.safetag&&(this.safetag=!1);"undefined"===typeof this.safetagInProgress&&(this.safetagInProgress=!1);"undefined"===typeof this.safetagParameter&&(this.safetagParameter={});"undefined"===typeof this.update&&(this.update=function(){});this.saveSendinfoArguments=[];this.safetagTimeoutStarted=!1;this.version=405;this.getJSON=function(a){if(a&&
"{"==a.charAt(0)&&"}"==a.charAt(a.length-1))try{return eval("("+a+")")}catch(b){}return null};this.parseJSON=function(a,b){for(var c in a){var d=c;if("object"==typeof a[d])"undefined"!=typeof this.jsonPara[d]&&"object"!=typeof this.config[this.jsonPara[d][0]]&&(this.config[this.jsonPara[d][0]]={}),this.parseJSON(a[d],d);else if(b){if(isNaN(parseInt(d))||500>parseInt(d))this.config[this.jsonPara[b][0]][d]=a[d]}else"undefined"!=typeof this.jsonPara[d]&&(this.config[this.jsonPara[d][0]]=a[d])}};this.getMappingParam=
function(a){var b=a.split(""),c,d,e;for(c=0;c<b.length;c++)if(!isNaN(parseInt(b[c]))){d=c;break}d?(b=a.substr(0,d),e=a.substr(d,a.length-1)):b=a;return{mapping:"undefined"!=typeof this.jsonPara[b]?this.jsonPara[b][0]:!1,index:e?e:!1}};this.getConfig=function(a){for(var b={},c=0;c<this.defaultAttribute.length;c++){var d=this.defaultAttribute[c];b[d]=a?!1:this[d]}return b};this.getRequestCounter=function(a,b){var c=0;"before"==b&&c++;return"link"==a?this.linkCounter+=c:"click"==a?this.clickCounter+=
c:"page"==a?this.pageCounter+=c:"heatmap"==a?this.heatmapCounter+=c:"form"==a?this.formCounter+=c:0};this.getPluginConfig=function(a,b){return{instance:this,mode:a,type:b,requestCounter:this.getRequestCounter(a,b)}};this.checkAsynchron=function(a,b,c,d){"undefined"!=typeof window[a]?b&&b(!0,c):0>=d?b&&b(!1,c):window.setTimeout(function(){c.checkAsynchron(a,b,c,d-100)},100)};this.loadAsynchron=function(a,b,c,d){this.include(a)&&this.checkAsynchron(b,c?c:!1,this,d?d:2E3)};this.include=function(a){if(!document.createElement)return!1;
var b=document.getElementsByTagName("head").item(0),c=document.createElement("script");c.setAttribute("language","javascript");c.setAttribute("type","text/javascript");c.setAttribute("src",a);b.appendChild(c);return!0};this.executePlugin=function(a){if(this.executePluginFunction&&"string"===typeof this.executePluginFunction){this.epf=!1;for(var b=this.executePluginFunction.split(";"),c=0;c<b.length;c++)b[c]&&"function"===typeof window[b[c]]&&(this.epf=window[b[c]],this.epf(a))}};this.indexOf=function(a,
b,c){return a.indexOf(b,c?c:0)};this.wtTypeof=function(a){return"undefined"!==typeof a?1:0};this.wtLength=function(a){return"undefined"!==typeof a?a.length:0};this.getAttribute=function(a,b){return"string"==typeof a.getAttribute(b)?a.getAttribute(b):"object"==typeof a.getAttribute(b)&&"object"==typeof a.attributes[b]&&null!=a.attributes[b]?a.attributes[b].nodeValue:""};this.getTimezone=function(){return Math.round(-1*((new Date).getTimezoneOffset()/60))};this.wtHref=function(){return"undefined"!==
typeof window.wtLocationHref?window.wtLocationHref:this.wtLocation().href};this.wtLocation=function(){var a=document.location;if(!document.layers&&document.getElementById)try{a=top.document.location}catch(b){a=document.location}else a=top.document.location;return a};this.checkBrowser=function(){this.isIE=this.indexOf(navigator.appName,"Microsoft")?!1:!0;this.isIE||(this.isOpera=this.indexOf(navigator.appName,"Opera")?!1:!0,this.isOpera||(this.isSafari=-1!=navigator.vendor.toLowerCase().indexOf("apple"),
this.isChrome=-1!=navigator.vendor.toLowerCase().indexOf("google"),this.isSafari||this.isChrome||(this.isFirefox=-1!=navigator.userAgent.toLowerCase().indexOf("firefox"))))};this.checkBrowser();this.url2contentId=function(a){if(!a)return"no_content";a=/\/\/(.*)/.exec(a);return 1>a.length?"no_content":a[1].split("?")[0].replace(/\./g,"_").replace(/\//g,".").replace(/\.{2,}/g,".").toLowerCase().split(";")[0]};this.contentId=g.contentId?g.contentId:this.url2contentId(document.location.href);this.registerEvent=
function(a,b,c){a.addEventListener?("webkitvisibilitychange"==b&&this.unregisterEvent(a,b,c),a.addEventListener(b,c,!1)):a.attachEvent&&("beforeunload"!=b&&"webkitvisibilitychange"!=b||this.unregisterEvent(a,b,c),a.attachEvent("on"+b,c))};this.unregisterEvent=function(a,b,c){a.removeEventListener?a.removeEventListener(b,c,!1):a.detachEvent&&a.detachEvent("on"+b,c)};this.maxlen=function(a,b){return a&&a.length>b?a.substring(0,b-1):a};this.wtEscape=function(a){try{return encodeURIComponent(a)}catch(b){return escape(a)}};
this.wtUnescape=function(a){try{return decodeURIComponent(a)}catch(b){return unescape(a)}};this.decrypt=function(a){var b="";if(a)try{b=this.wtUnescape(a.replace(/([0-9a-fA-F][0-9a-fA-F])/g,"%$1"))}catch(c){}return b};this.checkSC=function(a){if("string"!=typeof this.secureConfig)return!1;for(var b=this.secureConfig.split(";"),c=0;c<b.length;c++)if(b[c]==a)return!0;return!1};this.zeroPad=function(a,b){var c="000000000000"+a;return c.substring(c.length-b,c.length)};this.generateEid=function(){return"2"+
this.zeroPad(Math.floor((new Date).getTime()/1E3),10)+this.zeroPad(Math.floor(1E6*Math.random()),8)};this.getexpirydate=function(a){var b=new Date,c=b.getTime();b.setTime(c+6E4*a);return b.toUTCString()};this.setCookie=function(a,b,c){var d=location.hostname;-1==d.search(/^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$/)&&(d=location.hostname.split("."),d=d[d.length-2]+"."+d[d.length-1]);var e=!1;if(this.cookieDomain)for(var f=this.cookieDomain.split(";"),k=0;k<f.length;k++)if(-1!=location.hostname.indexOf(f[k])){d=
f[k];e=!0;break}a=e&&"undefined"!=typeof c&&c?a+"="+escape(b)+";domain="+d+";path=/;expires="+this.getexpirydate(c):e?a+"="+escape(b)+";path=/;domain="+d:3>d.split(".")[0].length&&"undefined"!=typeof c&&c?a+"="+escape(b)+";path=/;expires="+this.getexpirydate(c):3>d.split(".")[0].length?a+"="+escape(b)+";path=/":"undefined"!=typeof c&&c?a+"="+escape(b)+";domain="+d+";path=/;expires="+this.getexpirydate(c):a+"="+escape(b)+";path=/;domain="+d;document.cookie=a};this.getCookie=function(a){for(var b=document.cookie.split(";"),
c=0;c<b.length;c++){var d=b[c].substr(0,b[c].indexOf("=")),e=b[c].substr(b[c].indexOf("=")+1),d=d.replace(/^\s+|\s+$/g,"");if(d==a)return unescape(e)}return""};if(this.optOut=this.getCookie(this.optoutName)?!0:!1)this.deactivatePixel=!0;this.urlParam=function(a,b,c){if(!a||null===a||"undefined"===typeof a)return c;var d=[];0<a.indexOf("?")&&(d=a.split("?")[1].replace(/&amp;/g,"&").split("#")[0].split("&"));for(a=0;a<d.length;a++)if(0==d[a].indexOf(b+"="))return this.wtUnescape(d[a].substring(b.length+
1).replace(/\+/g,"%20"));return c};this.allUrlParam=function(a,b){if(this.mediaCodeFrames&&""!=this.mediaCodeFrames){for(var c=this.mediaCodeFrames.split(";"),d=0;d<c.length;d++){var e=!1;try{e=eval(c[d])}catch(f){}if(e&&(e!=top&&e.location)&&(e=this.urlParam(e.location.href,a,b),e!=b))return e}return b}c="";try{c=top.location.href}catch(k){c=document.location.href}return this.urlParam(c,a,b)};this.plugInArray=function(a,b){if("object"!=typeof a)return!1;for(var c=0;c<a.length;c++){var d=RegExp(a[c].toLowerCase(),
"g");if(-1!=b.toLowerCase().search(d))return a[c]}return!1};this.quicksend=function(a,b,c){if(!this.trackDomain||!this.trackId||this.deactivatePixel||this.deactivateRequest)this.deactivateRequest=!1;else{c||(c="wt"+(this.fileSuffix?".pl":""));"undefined"==typeof this.requestTimeout&&(this.requestTimeout=5);"1"==this.cookie&&(b="&eid="+this.eid+"&one="+(this.cookieOne?"1":"0")+"&fns="+(this.forceNewSession?"1":"0")+b);"1"!=this.cookie&&(this.wtTypeof(this.cookieEidTimeout)||this.wtTypeof(this.cookieSidTimeout))&&
(this.wtTypeof(this.cookieEidTimeout)&&""!=this.cookieEidTimeout&&(b="&cet="+this.cookieEidTimeout+b),this.wtTypeof(this.cookieSidTimeout)&&""!=this.cookieSidTimeout&&(b="&cst="+this.cookieSidTimeout+b));0<this.pixelSampling&&(b+="&ps="+this.pixelSampling);b="&tz="+this.getTimezone()+b;a="//"+this.trackDomain+"/"+this.trackId+"/"+c+"?p="+this.version+","+a+b+"&eor=1";if(!this.ignorePrerendering&&(this.isChrome&&"undefined"!=typeof document.webkitHidden)&&("object"!=typeof this.prerendering&&(this.prerendering=
[]),document.webkitHidden)){this.prerendering.push(a);var d=this;this.registerEvent(document,"webkitvisibilitychange",function(){d.sendPrerendering()});return}this.sendPixel(a,c);"hm"!=c&&"hm.pl"!=c&&(this.forceNewSession=this.cookieOne=!1,this.sentFullPixel=1)}};this.sendPrerendering=function(){if(!document.webkitHidden){for(var a=0;a<this.prerendering.length;a++)this.sendPixel(this.prerendering[a]);this.prerendering=[]}};this.sendPixel=function(a,b){a=this.maxlen(a,this.maxRequestLength);this.isFirefox?
this.sendPixelElement(a,b):this.sendPixelImage(a,b)};this.sendPixelImage=function(a,b){var c="https:"==document.location.protocol?"https:":"http:";this.forceHTTPS&&(c="https:");0!==a.search(/https:|http:/)&&(a=c+a);if("hm"==b||"hm.pl"==b)a+="&hm_ts="+(new Date).getTime();if("function"!==typeof d)var d=function(){return document.createElement("img")};if("undefined"===typeof e)var e=[];c=e.length;e[c]=new d;e[c].onload=function(){return!1};e[c].src=a};this.createPixelElement=function(a){a=a.replace(/'/g,
"%27");var b=document.createElement("div");b.style.width="0px";b.style.height="0px";b.style.backgroundImage="url('"+a+"')";window.setTimeout(function(){b.parentElement.removeChild(b)},5E3);return b};this.sendPixelElement=function(a,b){var c="https:"==document.location.protocol?"https:":"http:";this.forceHTTPS&&(c="https:");0!==a.search(/https:|http:/)&&(a=c+a);if("hm"==b||"hm.pl"==b)a+="&hm_ts="+(new Date).getTime();if("undefined"===typeof this.sendPixelObject||null===this.sendPixelObject)(c=document.getElementById("webtrekk-image"))&&
null!==c?this.sendPixelObject=c:(this.sendPixelObject=document.createElement("div"),this.sendPixelObject.id="webtrekk-image",this.sendPixelObject.style.width="0px",this.sendPixelObject.style.height="0px",this.sendPixelObject.style.overflow="hidden",c=null,"object"===typeof document.body?c=document.body:"object"===typeof document.getElementsByTagName("body")[0]&&(c=document.getElementsByTagName("body")[0]),c&&null!==c?c.appendChild(this.sendPixelObject):(this.sendPixelObject=null,this.sendPixelImage(a,
"wt"+(this.fileSuffix?".pl":""))));null!==this.sendPixelObject&&this.sendPixelObject.appendChild(this.createPixelElement(a))};this.checkCustomParameter=function(a,b){var c="";if("object"==typeof a)for(var d in a){var e=d;!isNaN(parseInt(e))&&(this.wtTypeof(a[e])&&"string"===typeof a[e]&&""!==a[e])&&(this.checkSC("custom")&&(a[e]=this.decrypt(a[e])),-1==this.paramFirst.indexOf(b+e+";")&&(c+="&"+b+e+"="+this.wtEscape(a[e])))}return c};this.send=function(a,b,c){if("link"==b||"click"==b)this.config.linkId=
a;this.config.contentId=this.config.contentId?this.config.contentId:this.contentId;var d=!b||"link"!=b&&"click"!=b?a?a:this.config.contentId:this.config.contentId;d||(d="no_content");a="";var d=this.wtEscape(d)+",1,",d=d+this.baseparams(),e=navigator.plugins.length,f="";if(0<e){for(var k=[],m=0;m<e;m++)navigator.plugins&&"Microsoft Internet Explorer"!=navigator.appName&&(f="Shockwave Flash"==navigator.plugins[m].name?navigator.plugins[m].description:navigator.plugins[m].name,(f=this.plugInArray(this.plugins,
f))&&!this.plugInArray(k,f)&&k.push(f));f=k.join("|")}if(this.paramFirst)for(e=this.paramFirst.split(";"),m=0;m<e.length;m++){var p=this.getMappingParam(e[m]),k=p.mapping,p=p.index;k&&(p?this.config[k]&&("undefined"!=typeof this.config[k][p]&&this.config[k][p])&&(a+="&"+e[m]+"="+this.wtEscape(this.config[k][p])):this.config[k]&&(a+="&"+e[m]+"="+this.wtEscape(this.config[k])))}if("string"===typeof c&&""!==c)for(var g=c.split(/;/),m=0;m<g.length;m++)this.wtTypeof(g[m])&&(c=g[m].split(/=/),this.checkSC("custom")&&
(c[1]=this.decrypt(c[1])),c[1]=this.wtEscape(c[1]),a+="&"+c[0]+"="+c[1]);else{this.wtEpEncoded=!1;c=this.checkCustomParameter(this.config.customParameter,"cp");e=this.checkCustomParameter(this.config.customSessionParameter,"cs");k=this.checkCustomParameter(this.config.customTimeParameter,"ce");this.config.couponValue&&(this.config.customEcommerceParameter||(this.config.customEcommerceParameter={}),this.config.customEcommerceParameter[563]=this.config.couponValue);p=this.checkCustomParameter(this.config.customEcommerceParameter,
"cb");this.config.orderValue&&-1==this.paramFirst.indexOf("ov;")&&(a=this.checkSC("order")?a+("&ov="+this.wtEscape(this.decrypt(this.config.orderValue))):a+("&ov="+this.wtEscape(this.config.orderValue)));this.config.currency&&-1==this.paramFirst.indexOf("cr;")&&(a=this.checkSC("order")?a+("&cr="+this.wtEscape(this.decrypt(this.config.currency))):a+("&cr="+this.wtEscape(this.config.currency)));this.config.orderId&&-1==this.paramFirst.indexOf("oi;")&&(a+="&oi="+this.wtEscape(this.config.orderId));this.config.product&&
(-1==this.paramFirst.indexOf("ba;")&&(a+="&ba="+this.wtEscape(this.config.product)),this.config.productCost&&-1==this.paramFirst.indexOf("co;")&&(a+="&co="+this.wtEscape(this.config.productCost)),this.config.productQuantity&&-1==this.paramFirst.indexOf("qn;")&&(a+="&qn="+this.wtEscape(this.config.productQuantity)),a+=this.checkCustomParameter(this.config.productCategory,"ca"),this.config.productStatus&&-1==this.paramFirst.indexOf("st;")&&(a+="&st="+this.wtEscape(this.config.productStatus)));m=s("wt_cd",
"(.*)");this.config.customerId||(this.config.customerId=m);this.config.customerId&&-1==this.paramFirst.indexOf("cd;")&&(a+="&cd="+this.wtEscape(this.config.customerId));a+=this.checkCustomParameter(this.config.crmCategory,"vc");!this.config.birthday&&(this.config.birthdayJ&&this.config.birthdayM&&this.config.birthdayD)&&(this.config.birthday=this.config.birthdayJ+this.config.birthdayM+this.config.birthdayD);this.config.telefon&&(this.config.telefon=this.config.telefon.replace(/\W|_/g,""));this.config.urmCategory||
(this.config.urmCategory={});this.config.urmCategory[700]=this.config.email;this.config.urmCategory[701]=this.config.emailRID;this.config.urmCategory[702]=this.config.emailOptin;this.config.urmCategory[703]=this.config.firstName;this.config.urmCategory[704]=this.config.lastName;this.config.urmCategory[705]=this.config.telefon;this.config.urmCategory[706]=this.config.gender;this.config.urmCategory[707]=this.config.birthday;this.config.urmCategory[708]=this.config.country;this.config.urmCategory[709]=
this.config.city;this.config.urmCategory[710]=this.config.postalCode;this.config.urmCategory[711]=this.config.street;this.config.urmCategory[712]=this.config.streetNumber;this.config.urmCategory[713]=this.config.validation;a+=this.checkCustomParameter(this.config.urmCategory,"uc");this.browserLang&&(a+="&la="+this.wtEscape(this.browserLang));a+=this.checkCustomParameter(this.config.contentGroup,"cg");var h="";this.config.campaignId&&(this.config.campaignId in this.sentCampaignIds?this.config.campaignId=
"ignore%3Dignore":this.sentCampaignIds[this.config.campaignId]=!0,-1==this.paramFirst.indexOf("mc;")&&(a+="&mc="+this.wtEscape(this.config.campaignId)),-1==this.paramFirst.indexOf("mca;")&&(a+="&mca="+(this.config.campaignAction?this.config.campaignAction.substring(0,1):"c")));h+=this.checkCustomParameter(this.config.customCampaignParameter,"cc");this.trackingSwitchMediaCode&&(a+="&tmc="+this.wtEscape(this.trackingSwitchMediaCode));this.trackingSwitchMediaCodeValue&&(a+="&tmcv="+this.wtEscape(this.trackingSwitchMediaCodeValue));
this.trackingSwitchMediaCodeTimestamp&&(a+="&tmct="+this.wtEscape(this.trackingSwitchMediaCodeTimestamp));"object"==typeof wt_ts&&"undefined"!=typeof wt_ts.trackingSwitchMediaCode&&(a+="&tmc="+this.wtEscape(wt_ts.trackingSwitchMediaCode));"object"==typeof wt_ts&&"undefined"!=typeof wt_ts.trackingSwitchMediaCodeValue&&(a+="&tmcv="+this.wtEscape(wt_ts.trackingSwitchMediaCodeValue));"object"==typeof wt_ts&&"undefined"!=typeof wt_ts.trackingSwitchMediaCodeTimestamp&&(a+="&tmct="+this.wtEscape(wt_ts.trackingSwitchMediaCodeTimestamp));
var l="";"undefined"!=typeof wt_vt&&(g=wt_vt);this.wtTypeof(g)||(g=this.urlParam(location.href,"wt_vt",!1));if(g)for(var n=this.getCookie("wt_vt").split(";"),m=0;m<n.length;m++)-1!=n[m].indexOf(g+"v")&&(l="&wt_vt="+n[m].split("t")[0].split("v")[1]);l&&(a+=l);this.config.internalSearch&&-1==this.paramFirst.indexOf("is;")&&(a+="&is="+this.wtEscape(this.maxlen(this.wtUnescape(this.config.internalSearch),255)));c&&(a+=c);h&&(a+=h);k&&(a+=k);p&&(a+=p);e&&(a+=e);this.wtTypeof(this.config.customSid)&&""!=
this.config.customSid&&(a+="&csid="+this.config.customSid);this.wtTypeof(this.config.customEid)&&""!=this.config.customEid&&(a+="&ceid="+this.config.customEid);this.wtTypeof(this.config.xwtip)&&""!=this.config.xwtip&&(a+="&X-WT-IP="+this.wtEscape(this.config.xwtip));this.wtTypeof(this.config.xwtua)&&""!=this.config.xwtua&&(a+="&X-WT-UA="+this.wtEscape(this.config.xwtua));this.wtTypeof(this.config.xwtrq)&&""!=this.config.xwtrq&&(a+="&X-WT-RQ="+this.wtEscape(this.config.xwtrq));this.wtTypeof(this.xwteid)&&
""!=this.xwteid&&(a+="&X-WT-EID="+this.wtEscape(this.xwteid),this.xwteid=!1);this.wtTypeof(this.config.xwtstt)&&""!=this.config.xwtstt&&(a+="&X-WT-STT="+this.wtEscape(this.config.xwtstt));!this.sentFullPixel&&this.firstVisitContact&&(a+="&fvc="+this.firstVisitContact);!this.sentFullPixel&&this.lastVisitContact&&(a+="&lvc="+this.lastVisitContact)}a+="&pu="+this.wtEscape(document.location.href.split("#")[0]);this.config.linkId&&this.config.customClickParameter&&(a+=this.checkCustomParameter(this.config.customClickParameter[this.config.linkId]?
this.config.customClickParameter[this.config.linkId]:this.config.customClickParameter,"ck"),this.ccParams=!1);this.config.xlc&&this.config.xlct&&(""!=this.config.xlc||""!=this.config.xlct)&&(g="",g=this.config.xlcv?this.getExtLifeCycles(this.config.xlc,this.config.xlct,this.config.xlcv):this.getExtLifeCycles(this.config.xlc,this.config.xlct),a+=g);this.config.contentId||this.config.linkId||(this.config.contentId=this.contentId,this.config.linkId="wt_ignore");this.config.linkId?(this.wtEp=a,this.wtEpEncoded=
!0,u("noForm",b)):("1"==this.cookie?this.cookieOne&&(a+="&np="+this.wtEscape(f)):a+="&np="+this.wtEscape(f),this.quicksend(d,a))};this.sendinfo_media=function(a,b,c,d,e,f,k,m){this.wtTypeof(wt_sendinfo_media)&&wt_sendinfo_media(a,b,c,d,e,f,k,m,this.unloadInstance)};this.getExtLifeCycles=function(a,b,c){for(var d="",e={},f=a.split("|"),k=0;k<f.length;k++){for(var m=f[k].split(";"),p=0;p<m.length;p++)d=0==p?d+this.wtEscape(m[p]):d+m[p],d+=";";d=d.substr(0,d.length-1);d+="|"}d=d.substr(0,d.length-1);
e.xlcl=this.wtEscape(a.split("|").length);e.xlct=this.wtEscape(b);"undefined"!=typeof c&&(e.xlcv=this.wtEscape(c));e.xlc=this.wtEscape(d);a="";for(var g in e)a+="&"+k+"="+e[g];return a};this.isOwnDomain=function(a){var b="";if(this.domain)if(0==this.domain.toUpperCase().indexOf("REGEXP:")){if(b=RegExp(this.domain.substring(7),"i"),b.test(this.getDomain(a)))return!0}else{b=this.domain.split(";");a=this.getDomain(a);for(var c=0;c<b.length;c++)if(a==b[c])return!0}return!1};this.getDomain=function(a){if("string"!=
typeof a)return"";a=this.wtUnescape(a);a=a.split("://")[1];var b=RegExp("^(?:[^/]+://)?([^/:]+)","g");return"undefined"!=typeof a&&(a=a.match(b),a[0])?a[0].toLowerCase():""};this.baseparams=function(){var a=screen.width+"x"+screen.height+","+("Netscape"!=navigator.appName?screen.colorDepth:screen.pixelDepth)+",",a=a+(!0==navigator.cookieEnabled?"1,":!1==navigator.cookieEnabled?"0,":-1!=document.cookie.indexOf("=")?"1,":"0,"),a=a+((new Date).getTime()+","),b="0",c=s("wt_ref","(.*)");f.framesetReferrer?
b=f.wtEscape(f.framesetReferrer):""!=f.getCookie("wt_ref")?(b=f.wtEscape(f.getCookie("wt_ref")),f.setCookie("wt_ref","",-3600)):c?b=f.wtEscape(c):0<document.referrer.length&&(b=f.wtEscape(document.referrer));f.sentFullPixel?b="2":f.isOwnDomain(b)&&(b="1");c=0;if(!document.layers&&document.getElementById)try{c=top.window.innerHeight}catch(d){}else c=top.window.innerHeight;if(!c)try{c=top.document.documentElement.clientHeight}catch(e){}if(!c)try{c=top.document.body.clientHeight}catch(t){}"undefined"==
typeof c&&(c=-1);var k;k=0;if(!document.layers&&document.getElementById)try{k=top.window.innerWidth}catch(m){}else k=top.window.innerWidth;if(!k)try{k=top.document.documentElement.clientWidth}catch(g){}if(!k)try{k=top.document.body.clientWidth}catch(h){}"undefined"==typeof k&&(k=-1);c&&c>screen.height&&(c=screen.height);k&&k>screen.width&&(k=screen.width);a=a+b+(","+k+"x"+c);return a+=","+(navigator.javaEnabled()?"1":"0")};this.getMediaCode=function(a){if(!a){if(!this.mediaCode)return;a=this.mediaCode}var b=
[];a=a.split(";");var c=0,d=[];this.mediaCodeValue&&(b=this.mediaCodeValue.split(";"));for(var e=0;e<a.length;e++){var f=this.allUrlParam(a[e],"");if(this.mediaCodeCookie){var k=!1,m=(this.trackId+"").split(",")[0],g=this.getCookie("wt_mcc_"+m),h;h=a[e]+"_"+f;for(var l=0,n=h.length,x=void 0,q=0;q<n;q++)x=h.charCodeAt(q),l=(l<<5)-l+x,l&=l;h=l>1E15-1?"0":l+"";-1===g.indexOf(","+h+",")&&f?(d.push(a[e]+this.wtEscape("="+f)),k=!0):c++;k&&(f="","eid"==this.mediaCodeCookie&&(f=2592E3),this.setCookie("wt_mcc_"+
m,g+","+h+",",f))}else"undefined"!==typeof b&&"undefined"!==typeof b[e]&&""!==b[e]?d.push(a[e]+this.wtEscape("="+b[e])):""!==f&&d.push(a[e]+this.wtEscape("="+f))}a.length===c&&0!==a.length&&d.push("ignore%3Dignore");this.config.campaignId=d.join(";")};this.searchContentIds=function(){var a=0,b=0;this.contentIds=this.wtEscape(this.contentId.split(";")[0]);do{a++;var c=this.urlParam(document.location.href,"wt_contentId"+a,!1);c&&(this.contentIds+="&wt_contentId"+a+"="+this.wtEscape(c),b++)}while(b>=
a)};var v=function(a){var b=f.reporturl;null!==a.match(/^(http[s]?:\/\/)?(report\d+|analytics)\.webtrekk\.(com|de).*$/)&&(b=a.split("/"),b.pop(),b=b.join("/"));return b};this.startHeatmapOrOverlay=function(a,b){this.searchContentIds();this.urlParam(this.wtHref(),"wt_reporter",!1)?this.reporturl=v(this.urlParam(this.wtHref(),"wt_reporter",!1)):this.getCookie("wt_overlayFrame")&&(this.reporturl=v(this.getCookie("wt_overlayFrame")));-1===this.reporturl.search(/http|https/)&&(this.reporturl=document.location.protocol+
"//"+this.reporturl);this.contentIds&&this.include(this.reporturl+"/"+a+".pl?wt_contentId="+this.contentIds+"&x="+(new Date).getTime())&&("heatmap"==a&&(-1!=navigator.userAgent.indexOf("MSIE 6")&&-1!=navigator.userAgent.indexOf("Windows NT 5.0"))&&alert("Click OK to start heatmap."),"complete"!=document.readyState?this.registerEvent(window,"load",b):b())};this.heatmapOn=0<=this.wtHref().indexOf("wt_heatmap=1");this.overlayOn=0<=this.wtHref().indexOf("wt_overlay=1")||0<=document.cookie.indexOf("wt_overlay=1");
0<=this.wtHref().indexOf("wt_overlay=0")&&(this.overlayOn=!1,this.setCookie("wt_overlay","",-1));this.heatmapTrackInit=function(){for(var a=!1,b=0;b<webtrekkHeatmapObjects.length;b++)this==webtrekkHeatmapObjects[b]&&(a=!0);!a&&(this.heatmap&&"1"==this.heatmap)&&(webtrekkHeatmapObjects.push(this),this.registerEvent(document,"mousedown",q),this.registerEvent(document,"touchstart",q))};this.heatmapTrackInit();this.heatmapOn&&!this.disableOverlayView&&this.startHeatmapOrOverlay("heatmap",r);this.overlayOn&&
!this.disableOverlayView&&(this.setCookie("wt_overlay","1"),this.startHeatmapOrOverlay("overlay",l));this.setPixelSampling=function(a){a||(a=this.pixelSampling);for(var b=this.trackId.split(",")[0],c=this.getCookie("wt3_sample").split(";"),d=!1,e=0;e<c.length;e++)-1!=this.indexOf(c[e],b+"|"+a)?d=!0:-1!=this.indexOf(c[e],b+"|")&&(c[e]="");e=6;this.cookieEidTimeout&&(e=this.cookieEidTimeout);d?(c=c.join(";"),this.setCookie("wt3_sample",c,43200*e)):(Math&&Math.random&&0==parseInt(Math.random()*a)?c.push(b+
"|"+a+"|1"):c.push(b+"|"+a+"|0"),this.setCookie("wt3_sample",c.join(";"),43200*e),c=this.getCookie("wt3_sample"));-1==this.indexOf(c,b+"|"+a+"|1")&&(this.deactivatePixel=!0)};this.pixelSampling&&!this.optOut&&this.setPixelSampling();this.firstParty=function(){for(var a=this.getCookie("wt3_sid").split(";"),b=this.getCookie("wt3_eid").split(";"),c=0===this.cookieEidTimeout?this.cookieEidTimeout:this.cookieEidTimeout?this.cookieEidTimeout:6,d=this.trackId.split(",")[0],e=!1,f=!1,k=!1,m=this.generateEid(),
g=0;g<a.length;g++)if(-1!=a[g].indexOf(d)){e=g;break}for(g=0;g<b.length;g++)if(-1!=b[g].indexOf(d+"|")){f=g;break}this.eid=s("wt_eid","^[0-9]{19}$");e||(a.push(d),f&&!this.eid&&(this.forceNewSession=!0));f?(this.eid&&(b[f]=d+"|"+this.eid,this.updateCookie=!0),-1==b[f].indexOf("#")&&(b[f]+="#"+m),k=b[f].replace(d+"|","").split("#")[1],this.eid=b[f].replace(d+"|","").split("#")[0],b[f]=b[f].replace(/#[0-9]{19}/g,"#"+m),this.updateCookie&&(c?this.setCookie("wt3_eid",b.join(";"),43200*c):this.setCookie("wt3_eid",
b.join(";")))):(this.eid||(this.eid=this.generateEid(),this.cookieOne=!0),b.push(d+"|"+this.eid+"#"+m),c?this.setCookie("wt3_eid",b.join(";"),43200*c):this.setCookie("wt3_eid",b.join(";")));this.setCookie("wt3_sid",a.join(";"));e||(this.firstVisitContact=n(this.eid),this.updateCookie&&(this.lastVisitContact=n(k?k:m)))};g=!1;for(h=0;h<webtrekkUnloadObjects.length;h++)this==webtrekkUnloadObjects[h]&&(g=!0);g||webtrekkUnloadObjects.push(this);var y=function(){var a=[],b;for(b in f.safetagParameter){var c=
b;if("executePluginFunction"==c)f.executePluginFunction+=f.safetagParameter[c],f.safetagParameter[c]="";else if("object"===typeof f.safetagParameter[c]){"object"!==typeof f[c]&&(f[c]={});for(var d in f.safetagParameter[c]){var e=d;f[c][e]=f.safetagParameter[c][e]}}else f[c]=f.safetagParameter[c],"form"!=c&&"linkTrack"!=c&&"heatmap"!=c||a.push(c)}for(b=0;b<a.length;b++)switch(a[b]){case "form":f.formTrackInstall();break;case "linkTrack":f.linkTrackInit();break;case "heatmap":f.heatmapTrackInit()}f.safetagParameter.pixel=
f},z=function(){f.safetagTimeoutStarted=!0;var a=(new Date).getTime()-f.startSafetagTimeoutDate;if(f.safetagInProgress&&a<f.safetag.timeout)window.setTimeout(function(){z()},5);else{f.safetagTimeoutStarted=!1;f.safetagInProgress=!1;a>f.safetag.timeout&&(f.xwtstt=f.safetag.timeout+"");for(a=0;a<f.saveSendinfoArguments.length;a++){var b=f.saveSendinfoArguments[a];f.sendinfo(b[0],b[1],b[2],b[3])}f.saveSendinfoArguments=[]}};this.sendinfo=function(a,b,c,d){c=c?c:"page";-1==location.href.indexOf("fb_xd_fragment")&&
(this.safetag&&y(),this.config="object"==typeof a?a:this.getConfig(),this.safetagInProgress?(a?f.saveSendinfoArguments.push([this.config,b,c,d]):f.saveSendinfoArguments.push([!1,b,c,d]),this.safetagTimeoutStarted||(this.startSafetagTimeoutDate=(new Date).getTime(),window.setTimeout(function(){z()},5))):(this.config.linkId&&(c="click",b||(b=this.config.linkId)),!1!=this.beforeSendinfoPixel?this.beforeSendinfoPixel():this.executePlugin(this.getPluginConfig(c?c:"page","before")),this.safetag&&y(),this.optOut||
this.deactivatePixel||("1"==this.cookie?this.firstParty():this.xwteid=s("wt_eid","^[0-9]{19}$")),this.config.campaignId||(!this.mediaCode||"page"!=c||this.config.linkId)||this.getMediaCode(),(""!=this.contentId||""!=b||document.layers)&&this.send(b,c,d),!1!=this.afterSendinfoPixel?this.afterSendinfoPixel():this.executePlugin(this.getPluginConfig(c?c:"page","after"))))};(function(a){var b=function(b,d){var e=this;e.item=d;e.href="undefined"!==typeof d.href?d.href:a.getAttribute(d,"href")?a.getAttribute(d,
"href"):"";e.linkIdByNameOrId=a.getAttribute(d,"name")?a.getAttribute(d,"name"):a.getAttribute(d,"id")?a.getAttribute(d,"id"):"";e.linkId="";e.action="link";e.isDownloadFile=!1;e.linktrackOut=!1;e.isInternalLink=function(){var b;if(a.linkTrackDownloads){b=e.href.split(".");b=b.pop();for(var c=a.linkTrackDownloads.split(";"),f=0;f<c.length;f++)if(c[f]==b){e.isDownloadFile=!0;break}}e.linktrackOut=a.domain&&!a.isOwnDomain(e.href);if(e.isDownloadFile||"_blank"===d.target)e.action="click";b=e.href;var c=
b.toLowerCase(),f=b.split("#")[0],g=document.location,h=e.item,l=a.getAttribute,n=l(h,"onclick"),q=l(h,"onmousedown"),h=l(h,"ontouchstart");b=a.noDelayLinkTrackAttribute?!!a.getAttribute(d,a.noDelayLinkTrackAttribute):!(b&&!(0===c.indexOf("javascript:")||0===c.indexOf("#")||"click"===e.action||f==g.href.split("#")[0]&&-1!==b.indexOf("#")||f==g.pathname.split("#")[0]&&-1!==b.indexOf("#")||n&&-1!==n.search(/return false[;]?$/)||q&&-1!==q.search(/return false[;]?$/)||h&&-1!==h.search(/return false[;]?$/)));
return b};e.getCCParams=function(){var b="";if(a.config.customClickParameter){var c="undefined"!==typeof a.config.customClickParameter[e.linkIdByNameOrId]?a.config.customClickParameter[e.linkIdByNameOrId]:!1;c||(c=a.config.customClickParameter);var d,f;for(f in c)d=f,!isNaN(parseInt(d))&&("string"===typeof c[d]&&c[d])&&(a.checkSC("custom")&&(c[d]=a.decrypt(c[d])),b+="&ck"+d+"="+a.wtEscape(c[d]))}return b};e.setJSONParams=function(){e.linkId||(e.linkId=a.getAttribute(d,a.linkTrackAttribute));null!==
a.getJSON(e.linkId)&&(a.parseJSON(a.getJSON(e.linkId)),e.linkId=a.config.linkId)};e.getLinkId=function(){e.linkId=a.getAttribute(d,a.linkTrackAttribute);e.setJSONParams();if("link"==a.linkTrack){var b=e.href.indexOf("//");e.href=0<=b?e.href.substr(b+2):e.href;a.linkTrackPattern&&(a.linkTrackReplace||(a.linkTrackReplace=""),e.href=e.href.replace(a.linkTrackPattern,a.linkTrackReplace));e.linkId=(e.linkId?e.linkId+".":"")+e.href.split("?")[0].split("#")[0].replace(/\//g,".");b=[];a.linkTrackParams&&
(b=a.linkTrackParams.replace(/;/g,",").split(","));for(var c=0;c<b.length;c++){var f=a.urlParam(e.href,b[c],"");f&&(e.linkId+="."+b[c]+"."+f)}}return e.linkId}};a.linkTrackObject=new function(){var c=this;c.triggerObjectName="__"+(new Date).getTime()+"_"+parseInt(1E3*Math.random());var d=function(b,e){var d=e[c.triggerObjectName];a.config=a.getConfig(!0);a.config.customClickParameter=a.customClickParameter;a.ccParams=d.getCCParams();var f=a.config.linkId=d.getLinkId();a.linktrackOut=d.linktrackOut;
a.sendinfo(a.config,f,d.action)},e=function(b){a.registerEvent(b,"click",function(e){if(e.which&&1==e.which||e.button&&1==e.button)a.delayLinkTrack&&("function"===typeof e.preventDefault&&!b[c.triggerObjectName].isInternalLink())&&(e.preventDefault(),window.setTimeout(function(){document.location.href=b.href},a.delayLinkTrackTime)),d(e,b)})};c.linkTrackInit=function(){if(a.linkTrack&&("link"==a.linkTrack||"standard"==a.linkTrack)){for(var d=!1,f=0;f<webtrekkLinktrackObjects.length;f++)a==webtrekkLinktrackObjects[f]&&
(d=!0);d||webtrekkLinktrackObjects.push(a);d=0;for(f=document.links.length;d<f;d++){var g=document.links[d],h=a.getAttribute(g,a.linkTrackAttribute),l=a.getAttribute(g,"href");(a.linkTrackIgnorePattern&&l&&-1==l.search(a.linkTrackIgnorePattern)||!a.linkTrackIgnorePattern)&&("undefined"===typeof g[c.triggerObjectName]&&(h||"link"==a.linkTrack))&&(g[c.triggerObjectName]=new b(c,g),e(g))}}};c.linkTrackInit()};a.linkTrackInstall=a.linkTrackObject.linkTrackInit;a.linkTrackInit=a.linkTrackObject.linkTrackInit})(f);
(function(a,b){var c=function(a,b){var c=null,f=b.type,g=a.getFormFieldName(b),h=a.getFormFieldValue(b);this.close=function(){null!==c&&(window.clearInterval(c),c=null,h=a.getFormFieldValue(b),a.formFieldData[g]=[f,h],a.formFieldDataPathAnalysis.push([g,f,h]))};this.start=function(){null===c&&(c=window.setInterval(function(){"undefined"!==typeof b&&(b&&null!==b)&&(h=a.getFormFieldValue(b),a.formFieldData[g]=[f,h])},50),delete a.formFieldDataUnused[g])}};a.formTrackObject=new function(b){var e=this,
f=a.wtTypeof(window.onbeforeunload)?"beforeunload":"unload";e.formObject=!1;e.formFocus=!1;e.formName=!1;e.form=a.form;e.formFieldData={};e.formFieldDataUnused={};e.formFieldDataPathAnalysis=[];e.triggerObjectName="__"+(new Date).getTime()+"_"+parseInt(1E3*Math.random());var k=function(a){return"hidden"!=a&&"button"!=a&&"image"!=a&&"reset"!=a&&"submit"!=a&&"fieldset"!=a},g=function(a){return"select-multiple"!=a&&"select-one"!=a&&"checkbox"!=a&&"radio"!=a},h=function(){l();if(e.formObject){var g=a.getAttribute(e.formObject,
a.formAttribute);e.formName=g?g:a.contentId.split(";")[0];for(var g=0,h=e.formObject.elements,m=h.length;g<m;g++){var p=h[g],w=e.getFormFieldName(p);k(p.type)&&(w&&null!==w)&&(e.formFieldData[w]=[p.type,e.getFormFieldValue(p)],e.formFieldDataUnused[w]=[p.type,e.getFormFieldValue(p)],function(b){a.registerEvent(b,"focus",function(){k(b.type)&&e.formObject&&(e.formFocus=b,e.formFocus[e.triggerObjectName]=new c(e,e.formFocus),e.formFocus[e.triggerObjectName].start())});a.registerEvent(b,"blur",function(){k(b.type)&&
e.formObject&&e.formFocus&&e.formFocus&&"undefined"!==typeof e.formFocus[e.triggerObjectName]&&e.formFocus[e.triggerObjectName].close()})}(p))}a.registerEvent(e.formObject,"submit",n);a.registerEvent(window,f,b)}},l=function(){if(e.form&&!e.formObject)for(var b=document.forms,c=0,d=b.length;c<d;c++){var f=b[c];if(a.wtTypeof(f.elements.wt_form)){e.formObject=f;break}}},n=function(b){!e.form||b.target!==e.formObject&&b.srcElement!==e.formObject||(a.formSubmit=!0)},q=function(b){var c=[];a.formFullContent&&
(c=a.formFullContent.split(";"));if(a.formAnonymous||g(b.type)){for(var d=0;d<c.length;d++)if(c[d]==e.getFormFieldName(b))return!1;return!0}return!1},r=function(b,c){c||(c=b);var e=a.getAttribute(c,a.formValueAttribute).replace(/[\.|;|\|]/g,"_");return g(b.type)?a.maxlen(a.wtUnescape(c.value),110):q(b)?"anon":a.maxlen(a.wtUnescape(e),110)},s=function(a,b,c,d){var f=a.replace(/[\.|;|\|]/g,"_")+".",f=f+(b+"|")+(c+"|");return f=d?f+"0":f+(e.formFocus&&e.getFormFieldName(e.formFocus)===a?"1":"0")},v=
function(a,b){for(var c=[],e=0,d=b.length;e<d;e++)if("undefined"!==typeof a[b[e]])if("select-multiple"==a[b[e]][0])for(var f=a[b[e]][1].split("|"),g=0,k=f.length;g<k;g++)c.push(s(b[e],a[b[e]][0],f[g]));else c.push(s(b[e],a[b[e]][0],a[b[e]][1]));return c.join(";")},u=function(){if(!e.formObject)return"";var b=[],c;c=[];if(a.wtTypeof(e.formObject.elements.wt_fields)){var d=e.formObject.elements.wt_fields.value;d&&(c=d.split(";"))}if(0>=c.length)for(var f in e.formFieldData)d=f,"string"===typeof d&&
d&&c.push(d);f=!1;if(a.formPathAnalysis){(d=v(e.formFieldDataUnused,c))&&b.push(d);for(var d=0,g=e.formFieldDataPathAnalysis,k=g.length;d<k;d++){var h;a:{h=0;for(var m=c.length;h<m;h++)if(g[d][0]===c[h]){h=!0;break a}h=!1}h&&(f=!0,b.push(s(g[d][0],g[d][1],g[d][2],!0)))}f&&(c=b[b.length-1],c=c.substr(0,c.length-1),b[b.length-1]=c+"1")}else return v(e.formFieldData,c);return b.join(";")};e.getFormFieldName=function(b){var c=b.name;a.formFieldAttribute&&(c="",(b=a.getAttribute(b,a.formFieldAttribute))||
null!==b)&&(c=b);return c};e.getFormFieldValue=function(b){var c=b.type,d="";if("select-multiple"==c){for(var d=[],f=0,g=b.options,k=g.length;f<k;f++)g[f].selected&&d.push(r(b,g[f]));0>=d.length&&d.push("empty");d=d.join("|")}else"select-one"==c?(d="",-1!==b.selectedIndex&&((d=r(b,b.options[b.selectedIndex]))&&null!==d||(d="empty"))):"checkbox"==c||"radio"==c?(d="",b.checked?(d=r(b))||(d="checked"):d="empty"):"hidden"!=c&&("button"!=c&&"image"!=c&&"reset"!=c&&"submit"!=c)&&(f=(d=r(b))?"filled_out":
"empty",q(b)||(f=d),g=e.getFormFieldName(b),"undefined"!==typeof a.formFieldDefaultValue[g]&&a.formFieldDefaultValue[g]==d&&"empty"!==f&&(f="empty"),f&&null!==f||(f="empty"),d=f);return q(b)&&"select-multiple"!==c&&"empty"!==d&&"filled_out"!==d?"anon":d};e.formTrackInstall=function(a){e.formObject=a?a:document.forms[0]?document.forms[0]:!1;e.formObject&&(e.form="1",h())};e.getFormTrackingData=function(){var b="";if(e.formObject){var c=u();c&&(b+="&fn="+a.wtEscape(e.formName)+"|"+(a.formSubmit?"1":
"0"),b+="&ft="+a.wtEscape(c));a.formSubmit=!1;e.formName=!1;e.formObject=!1;e.formFocus=!1;e.formFieldData={};e.formFieldDataUnused={};e.formFieldDataPathAnalysis=[]}return b};e.sendFormRequest=function(){!1!==a.beforeUnloadPixel?a.beforeUnloadPixel():a.executePlugin(a.getPluginConfig("form","before"));var b=e.getFormTrackingData();b&&a.quicksend(a.wtEscape(a.contentId.split(";")[0])+",1,"+a.baseparams(),b);!1!==a.afterUnloadPixel?a.afterUnloadPixel():a.executePlugin(a.getPluginConfig("form","after"))};
e.form&&"1"==e.form&&h()}(b);a.formTrackInstall=a.formTrackObject.formTrackInstall;a.formTrackInit=a.formTrackObject.formTrackInstall;a.sendFormRequest=a.formTrackObject.sendFormRequest;a.checkFormTracking=a.formTrackObject.getFormTrackingData})(f,function(){u("form")});(function(a){a.cookieManager=function(a,c,d){this.name=a;this.keySeperator="~";this.fieldSeparator="#";this.durationSeperator="|";this.found=!1;this.expires=c?c:!1;this.accessPath=d?d:"/";this.rawValue="";this.fields=[];this.fieldsDuration=
[];var e=function(a){try{return decodeURIComponent(a)}catch(b){return unescape(a)}},f=function(a){try{return encodeURIComponent(a)}catch(b){return escape(a)}};this.read=function(){this.rawValue=null;this.found=!1;for(var a=document.cookie.split(";"),b=0;b<a.length;b++){var c=a[b].substr(0,a[b].indexOf("=")),d=a[b].substr(a[b].indexOf("=")+1),c=c.replace(/^\s+|\s+$/g,"");c==this.name&&(this.rawValue=d,this.found=!0)}if(null!==this.rawValue){a=b=0;do a=this.rawValue.indexOf(this.fieldSeparator,b),-1!=
a&&(b=this.rawValue.substring(b,a).split(this.durationSeperator),c=b[0].split(this.keySeperator),this.fields[c[0]]=e(c[1]),this.fieldsDuration[c[0]]=parseInt(e(b[1])),b=a+1);while(-1!==a&&a!==this.rawValue.length-1)}return this.found};this.getSize=function(){var a=(new Date).getTime(),b="",c;for(c in this.fields){var d=c+"";this.fieldsDuration[d]>=a&&(b+=f(d)+this.keySeperator+f(this.fields[d])+this.durationSeperator+f(this.fieldsDuration[d])+this.fieldSeparator)}return b.length};this.write=function(){var a=
(new Date).getTime(),b=!0,c=this.name+"=",d;for(d in this.fields){var e=d+"";this.fieldsDuration[e]>=a&&(c+=f(e)+this.keySeperator+f(this.fields[e])+this.durationSeperator+f(this.fieldsDuration[e])+this.fieldSeparator,b=!1)}a=b?-99999:this.expires;""!==a&&"number"===typeof a&&(b=new Date,b.setTime((new Date).getTime()+864E5*a),c+="; expires="+b.toGMTString());null!==this.accessPath&&(c+="; PATH="+this.accessPath);a=location.hostname;-1==a.search(/^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$/)&&
(a=location.hostname.split("."),a=a[a.length-2]+"."+a[a.length-1]);document.cookie=c+("; DOMAIN="+a)};this.remove=function(){this.expires=-10;this.write();return this.read()};this.get=function(a){var b=(new Date).getTime();return this.fieldsDuration[a]>=b?this.fields[a]:""};this.set=function(a,b,c,d,e){c||(c=31536E3);d||(d="last");var f=(new Date).getTime();if("first"==d&&""!==this.fields[a]&&null!==this.fields[a]&&this.fieldsDuration[a]>=f)return this.fields[a];this.fields[a]=b;this.fieldsDuration[a]=
f+1E3*parseInt(c);e||this.write();return b};this.prepare=function(a,b,c,d){return this.set(a,b,c,d,!0)};this.read()}})(f)};
if("object"===typeof webtrekkConfig&&"object"===typeof webtrekkConfig.safetag&&-1===document.cookie.indexOf("wt_r=1")){var wts=wts||[],wt_safetagConfig={async:"undefined"!==typeof webtrekkConfig.safetag.async?webtrekkConfig.safetag.async:!0,timeout:"undefined"!==typeof webtrekkConfig.safetag.timeout?webtrekkConfig.safetag.timeout:2E3,safetagDomain:"undefined"!==typeof webtrekkConfig.safetag.safetagDomain?webtrekkConfig.safetag.safetagDomain:!1,safetagId:"undefined"!==typeof webtrekkConfig.safetag.safetagId?
webtrekkConfig.safetag.safetagId:!1,customDomain:"undefined"!==typeof webtrekkConfig.safetag.customDomain?webtrekkConfig.safetag.customDomain:!1,customPath:"undefined"!==typeof webtrekkConfig.safetag.customPath?webtrekkConfig.safetag.customPath:!1,option:"undefined"!==typeof webtrekkConfig.safetag.option?webtrekkConfig.safetag.option:{}};(function(g,u){var q=function(g){try{return encodeURIComponent(g)}catch(f){return escape(g)}},r=document.location.protocol;if("http:"==r||"https:"==r){var l="";g.customDomain&&
g.customPath?l=r+"//"+g.customDomain+"/"+g.customPath:g.safetagDomain&&g.safetagId&&(l=r+"//"+g.safetagDomain+"/resp/api/get/"+g.safetagId+"?url="+q(document.location.href));if(l){for(var s in g.option)l+="&"+s+"="+q(g.option[s]);webtrekkV3.prototype.safetag=g;webtrekkV3.prototype.safetagInProgress=!0;webtrekkV3.prototype.safetagParameter={};webtrekkV3.prototype.update=function(){};window.wts=u;window.safetagLoaderHandler=function(g,f){if(g&&f)if("onerror"==f)webtrekkV3.prototype.safetagInProgress=
!1;else if("onload"==f){if("undefined"!==typeof wt_r&&!isNaN(wt_r)){var l=new Date;document.cookie="wt_r=1;path=/;expires="+l.toUTCString(l.setTime(l.getTime()+1E3*parseInt(wt_r)))}webtrekkV3.prototype.safetagInProgress=!1}else"onreadystatechange"!=f||"loaded"!=g.readyState&&"complete"!=g.readyState||(g.onreadystatechange=null,g.onload(n))};if(g.async||"complete"===document.readyState||"interactive"===document.readyState){var q=document.getElementsByTagName("script")[0],n=document.createElement("script");
n.async=g.async;n.type="text/javascript";n.onerror=function(){safetagLoaderHandler(n,"onerror")};n.onload=function(){safetagLoaderHandler(n,"onload")};n.onreadystatechange=function(){safetagLoaderHandler(n,"onreadystatechange")};n.src=l;q.parentNode.insertBefore(n,q)}else l='<script type="text/javascript" src="'+l+'" onerror="safetagLoaderHandler(this,\'onerror\')"',l+=" onload=\"safetagLoaderHandler(this,'onload')\"",l+=" onreadystatechange=\"safetagLoaderHandler(this,'onreadystatechange')\"",l+=
">\x3c/script>",document.write(l)}}})(wt_safetagConfig,wts)};

var wt_socialMedia = function(conf) {
	if(conf.mode == "page" && conf.type == "before" && conf.requestCounter == 1){
		if(typeof(window.wt_instance) == "undefined"){window.wt_instance = [];}window.wt_instance.push(this);var ins = window.wt_instance[(window.wt_instance.length - 1)];try{if(FB && FB.Event && FB.Event.subscribe){FB.Event.subscribe('edge.create',function(targetUrl){ins.send("social media button","click","ck550=facebook - i like");});}}catch(e){};try{if(FB && FB.Event && FB.Event.subscribe){FB.Event.subscribe('edge.remove',function(targetUrl){ins.send("social media button","click","ck550=facebook - unlike");});}}catch(e){};try{if(FB && FB.Event && FB.Event.subscribe){FB.Event.subscribe('message.send',function(targetUrl){ins.send("social media button","click","ck550=facebook - send");});}}catch(e){};try{if(twttr && twttr.events && twttr.events.bind){twttr.events.bind('tweet',function(event){if(event){ins.send("social media button","click","ck550=twitter - tweet");}});}}catch(e){};this.googlePlusone = function(a){for(var i = 0; i < window.wt_instance.length; i++){if(typeof(window.wt_instance[i].sendGooglePlusone) == "undefined"){window.wt_instance[i].send("social media button","click","ck550=google - plus one");window.wt_instance[i].sendGooglePlusone = true;}}};window.wt_googlePlusone = this.googlePlusone;
	}
};
 
function wt_scrollposition(conf) {
	if(conf.mode == "page" && conf.type == "after" && conf.requestCounter == 1) {
		var instance = this;var event = ((this.wtTypeof(window.onbeforeunload)) ? "beforeunload" : "unload");var de = document.documentElement;var scrollPosition = window.scrollY+window.innerHeight || self.scrollY+self.innerHeight || (de&&de.scrollTop+de.clientHeight) || document.body.scrollTop+document.body.clientHeight;this.registerEvent(window, 'scroll', function () {var Y = window.scrollY+window.innerHeight || self.scrollY+self.innerHeight || (de&&de.scrollTop+de.clientHeight) || document.body.scrollTop+document.body.clientHeight;if (Y > scrollPosition) {scrollPosition = Y;}});this.registerEvent(window, event, function () {var YMax = window.innerHeight+window.scrollMaxY || self.innerHeight+self.scrollMaxY || (de&&de.scrollHeight) || document.body.offsetHeight;scrollPosition = Math.round(scrollPosition / YMax * 100);if(scrollPosition > 100) {scrollPosition = 100;}for(;;) {if(scrollPosition % 5 != 0) {scrollPosition++;}else {break;}}instance.sendinfo({linkId : "wt_ignore",customClickParameter : {"540" : "" + scrollPosition}});});
	}
};
$(function(){
//bookmarks: click the bookmark button to open productshare-layer
	$('#tool-shares').click(function(){
		var shareLayer = $(this).parent().find('.share-layer');		
		if (shareLayer.is(":visible")) {
			shareLayer.hide();
		} else {
			shareLayer.css({
				left :  30,
				top :  41
			}).show();
		}
		return false;
	}).bind('closeMe', function() {
		// Event to close the Layer
		$(this).parent().find('.share-layer').hide();
	});
});

/**
 * Tracking related functionality
 * 
 * Preconditions: "trackedAffiliates" and "generalCookieDomain" must be defined by javascript written by the server (BasePage) 
 * Keep in sync: cda, frontend, checkout, productpage (TODO)
 */
function getQueryVariable(variable) {
	var query = window.location.search.substring(1);
	var vars = query.split("&");
	for (var i=0;i<vars.length;i++) {
		var pair = vars[i].split("=");
		if(pair[0] == variable){return pair[1];}
	}
	return (false);
}

function trackAffiliate(affiliationKey) {
	$.cookie('trkaff', affiliationKey, { domain : generalCookieDomain, path: '/', expires: 30});
}

$(document).ready(function() {
	var wtmcValue = getQueryVariable('wt_mc');
	if(wtmcValue) {
		// media code found, check if it should be tracked
		if(typeof trackedAffiliates !== 'undefined') {
			wtmcValue = wtmcValue.replace('%20', ' ').replace('+', ' ');
			for (var i=0; i < trackedAffiliates.length; i++) {
				if(wtmcValue.indexOf(trackedAffiliates[i]) > -1) {
					trackAffiliate(trackedAffiliates[i]);
					break;
				}
			}
		}
	}
});$(document).ready(function() {
		if ($('.cms-career').length > 0) {
		
			function renderJobData(dataArray, fetchAll, isResetFilterVisible) {
				$('#' + targetElementId).empty();
				$('#careersMore').hide();

				var jobList = $("<ul id='jobList' class='cms-career-jobs'></ul>");
				jobList.appendTo($('#' + targetElementId))

				var locationFilter = $('#locations').val();
				var businessareasFilter = $('#businessareas').val();
				var noOfJobs = 0;

				for (var i = 0; i < dataArray.length; i++) {

					if (locationFilter != allOption && dataArray[i].country.value != locationFilter) {
						continue;
					}

					if (businessareasFilter != allOption && dataArray[i].filter1.value != businessareasFilter) {
						continue;
					}

					var linkUrl = dataArray[i].deeplinkJobUrl.value;
					var jobListElement = $('<li id="jobListElement' + i + '" class="cms-career-job" onclick="window.open(\'' + linkUrl + '\', \'_blank\');return false;"></li>');
					jobListElement.appendTo(jobList);

					var linkText = dataArray[i].extTitle.value;
					var id = dataArray[i].id.value;
					var linkelement = $("<h3><a href='" + linkUrl + "' target='_blank'>" + linkText + "</a></h3>");
					linkelement.appendTo($('#' + "jobListElement" + i));

					var businessArea = dataArray[i].filter1.value;
					var businessAreaElement = $("<p class='area'>" + businessArea + "</p>");
					businessAreaElement.appendTo($('#' + "jobListElement" + i));

					var location = dataArray[i].country.value;
					var city = dataArray[i].mfield1.value;
					var locationElement = $("<p class='location'>" + city + ', ' + mapName(location, countryNames) + "</p>");
					locationElement.appendTo($('#' + "jobListElement" + i));

					if (isResetFilterVisible) {
						$('#resetFilters').show();
					}

					noOfJobs++;
					if (!fetchAll && noOfJobs >= noOfJobsToBeDisplayed) {
						$('#careersMore').show();
						break;
					}
					
				}
				if(!fetchAll && noOfJobs < noOfJobsToBeDisplayed) {
					$('#careersMore').hide();
				}
			}

			function clearFilters() {
				$('#locations').val(allOption);
				$('#businessareas').val(allOption);
			}

			function mapName(key, mapping) {
				if(mapping == null) {
					return key;
				} else if(!isEmpty(mapping[key])) {
					return mapping[key];
				} else {
					return key;
				}
			}

			var dataArray;
			var isResetFilterVisible = false;

			var targetElementId = 'jobRequisitions';
			var allOption = "all";
			var noOfJobsToBeDisplayed = 10;

			var hasOwnProperty = Object.prototype.hasOwnProperty;

			function isEmpty(obj) {

				// null and undefined are "empty"
				if (obj == null)
					return true;

				// Assume if it has a length property with a non-zero
				// value
				// that that property is correct.
				if (obj.length > 0)
					return false;
				if (obj.length === 0)
					return true;

				// Otherwise, does it have any properties of its own?
				// Note that this doesn't handle
				// toString and valueOf enumeration bugs in IE < 9
				for ( var key in obj) {
					if (hasOwnProperty.call(obj, key))
						return false;
				}

				return true;
			}

			function populateFilters(dataArray) {

				var locationArray = [];
				var businessAreaArray = [];

				for (var i = 0; i < dataArray.length; i++) {
					var location = dataArray[i].country.value;
					locationArray.push(location);

					var businessArea = dataArray[i].filter1.value;
					businessAreaArray.push(businessArea);
				}

				Array.prototype.contains = function(v) {
					for (var i = 0; i < this.length; i++) {
						if (this[i] === v)
							return true;
					}
					return false;
				};

				Array.prototype.unique = function() {
					var arr = [];
					for (var i = 0; i < this.length; i++) {
						if (!arr.contains(this[i])) {
							arr.push(this[i]);
						}
					}
					return arr;
				}

				function getUniqueSortedArray(unsortedArray) {
					return unsortedArray.unique().sort();
				}

				function sortByMapping(arr, mapping) {
					var tempArr = [];
					for (var i = 0; i < arr.length; i++) {
						tempArr.push({key: arr[i], value: mapName(arr[i], mapping)});
					}
					tempArr = tempArr.sort(function(a, b) {return a.value.localeCompare(b.value);})
					var result = [];
					for (var i = 0; i < arr.length; i++) {
						result.push(tempArr[i].key);
					}
					return result;
				}

				function createOption(arr, elemName, mapping) {
					if(mapping != null) {
						arr = sortByMapping(arr, mapping);
					}
					$.each(arr, function(idx, elem) {
						$('#' + elemName).append($('<option value="' + elem + '">'+ mapName(elem, mapping) + '</option>'))
					})
				}

				createOption(getUniqueSortedArray(locationArray), "locations", countryNames)
				createOption(getUniqueSortedArray(businessAreaArray), "businessareas")
			}


			var careers = (function() {
				
				var handleChange = function() {
					isResetFilterVisible = true;
					renderJobData(dataArray, false, isResetFilterVisible);
				}

				return {
					handleResponse : function(data) {
						dataArray = data.sfobject;
						if (dataArray.length === 0) {
							$('#nojobs').show();
							$('#positionheader').hide();
						} else {
							var openPositionsCount = dataArray.length;
							
							var counterHtml = ' <span class="counter">'+ openPositionsCount +'</span>';
							
							var header1 = $('.cms-career h1').text();
							var header2 = $('.cms-career .section-header h2').text();
							
							$('.cms-career h1').html(header1 + counterHtml);
							$('.cms-career .section-header h2').html(header2 + counterHtml);

							populateFilters(dataArray);
							renderJobData(dataArray, false, isResetFilterVisible);

							$('#locations').on('change', handleChange);

							$('#businessareas').on('change', handleChange);

							$('#showMore').on('click',
									function(e) {
										$('#careersMore').hide();
										renderJobData(dataArray, true, isResetFilterVisible);
										e.preventDefault();
									});

							$('#resetFilters').on('click',
									function(e) {
										$('#resetFilters').hide();
										clearFilters();
										isResetFilterVisible = false;
										renderJobData(dataArray, false, isResetFilterVisible);
										e.preventDefault();
									});
							
							$('.cms-career-content').show();
						}
					}
				}
			})();

			function crossDomainAjax(url, successCallback) {
				// IE8 & 9 only Cross domain JSON GET request
				if ('XDomainRequest' in window && window.XDomainRequest !== null) {
					var xdr = new XDomainRequest(); // Use Microsoft XDR
					xdr.onload = function() {
						var dom = new ActiveXObject('Microsoft.XMLDOM'),
								JSON = $.parseJSON(xdr.responseText);

						dom.async = false;

						if (JSON == null || typeof (JSON) == 'undefined') {
							JSON = $.parseJSON(data.firstChild.textContent);
						}

						successCallback(JSON); // internal function
					};

					xdr.onerror = function() {
						_result = false;
					};

					xdr.onprogress = function() {};

					xdr.open('get', url);
					xdr.send();
				}

				// IE7 and lower can't do cross domain
				else if (navigator.userAgent.indexOf('MSIE') != -1 &&
						parseInt(navigator.userAgent.match(/MSIE ([\d.]+)/)[1], 10) < 8) {
					return false;
				}

				// Do normal jQuery AJAX for everything else
				else {
					$.ajax({
						url: url,
						cache: false,
						dataType: 'json',
						type: 'GET',
						async: false, // must be set to false
						success: function(data, success) {
							successCallback(data);
						}
					});
				}
			}

			var config = {
				"companyId" : "CC1C2DE9-EBC2-5C65-A253-83097B488441", // Server
				"locale" : "en_US;de_DE;nl_NL;en_GB",
				"sort" : [ {
					"filter1.value" : 1
				}, {
					"extTitle.value" : 1
				}]
			};
			$.support.cors = true;
			crossDomainAjax('//services.abayoo.de/queryJobs.php?'+$.param(config), careers.handleResponse);
		}
});
/* WIDGET TABS/PROCESS
----------------------------------------------------*/
$(function(e) {
	// convert DL-list to tab panel
	$('dl.cms-tab-panel').each(function(i) {
		$(this).prepend('<ul class="cms-tab-pane cms-cols-' + $('dt', this).length + '"/>');
		$('dd', this).each(function(j) {
			$(this).addClass('cms-tab-content');
			$(this).attr('id', 'tab-content-' + i + '-' + j)
			if (j > 0) {
				$(this).hide();
			}
		});
		$('dt', this).each(function(k) {
			var dataHref = '';
			var target = $(this).attr('data-href');
			if (typeof target !== 'undefined' && target !== false) {
				dataHref = 'data-href="' + target + '"';
			}
			$(this).parent('dl').find('ul.cms-tab-pane').append('<li class="tab-li" data-rel="tab-content-' + i + '-' + k + '" ' + dataHref + '>' + $(this).html() + '<b class="indicator"></b></li>');
		});
		if ($(this).hasClass('cms-process-visual')) {
			var maxHeight = 0;
			$('.tab-li', this).each(function(l) {
				if ($(this).height() > maxHeight)
					maxHeight = $(this).height();
			});
			$('.tab-li', this).css('height', maxHeight)
		}
		// initialize the tab panel
		$('li:first-child', this).addClass('active');
		$('dt', this).first().addClass('active');
		$('.tab-content:first-child', this).show();
	});

	// tab panel behaviour hover
	$('.cms-tab-pane > li').on('mouseenter', function() {
		$(this).siblings('li').removeClass('active');
		$(this).parent('ul').siblings('dd').hide();
		$('#' + $(this).attr('data-rel')).fadeIn();
		$(this).addClass('active')
	});	
	// tab panel behaviour click
	$('.cms-tab-pane > li').on('click', function() {
		if (!$(this).hasClass('active')) {
			$(this).siblings('li').removeClass('active');
			$(this).parent('ul').siblings('dd').hide();
			$('#' + $(this).attr('data-rel')).fadeIn();
			$(this).addClass('active')
		} else {
			var target = $(this).attr('data-href');
			if (typeof target !== 'undefined' && target !== false) {
				document.location = target;
			}
		}
	});
	// tab panel behaviour on small screens
	$('.cms-tab-panel > dt').on('click', function() {
		if (!$(this).hasClass('active')) {
			$(this).siblings('dt').removeClass('active');
			$(this).siblings('dd').slideUp();
			$(this).next('dd').slideDown();
			$(this).addClass('active')
		} else {
			var target = $(this).attr('data-href');
			if (typeof target !== 'undefined' && target !== false) {
				document.location = target;
			}
		}
	});
});

/* WIDGET TABS/PROCESS post-processing
----------------------------------------------------*/

/* TREE NAV
----------------------------------------------------*/
$(function() {
	
	// determine nodes with children
	$('.tree-nav li').each(function(i) {
		if ($('ol', this).length == 0) {
			$(this).addClass('last-node');
		} else {
			$(this).addClass('has-children');
		}

	});
	// assign levels to li
	$('.tree-nav .nav > li').addClass('level-1');

	// Marking active links Vertical Channel Navigation V2
	
	if ($('.tree-nav.tree-nav-v2 a').length >0){
	
		var highlightableUrls = getHighlightableUrls();
		var markedCurrentLinkInNavigation = false;
		for(var i=0; i<highlightableUrls.length; i++) {
			if(!markedCurrentLinkInNavigation) {
				$('.tree-nav.tree-nav-v2 a').each(function() {
					if(!markedCurrentLinkInNavigation && $(this).attr('href') == highlightableUrls[i]) {
						$(this).closest('li').addClass('current');
						markedCurrentLinkInNavigation = true;
					}
				});
			}
		}
		
		// initially open all ol that have a current li nested somewhere
		$('.tree-nav li.current').parents("ol").show();
		$('.tree-nav li.current').children("ol").show();
		$('.tree-nav li.current > .row > .columns').children("ol").show();

		// If a navigation with expander was expanded and has the expand-button
		// instead of the collapse-button than change it.
		if($('.hide-nav').is(':visible') && $('.show-nav').is(':visible')) {
			$('.show-nav').hide();
			$('.collapse-nav').css('display','block');
		}
	}

});

function getHighlightableUrls() {
	var urls = new Array();
	urls.push(document.location.pathname); // firstly: current url
	if (typeof cmContextUrl !== "undefined") {
		urls.push(cmContextUrl); // secondly: context/parent url from CM
	}
	var current = document.location.pathname;
	while(current.lastIndexOf('/') > 1) {
		current = current.substring(0, current.lastIndexOf('/'))
		if($.inArray(current, urls) == -1) {
			urls.push(current);
		}
	}
	if (typeof cmContextUrl !== "undefined") {
		var current = cmContextUrl;
		while(current.lastIndexOf('/') > 1) {
			current = current.substring(0, current.lastIndexOf('/'))
			if($.inArray(current, urls) == -1) {
				urls.push(current);
			}
		}
	}
	return urls;
}if(typeof angular !== 'undefined') {
    var shopApp = angular.module('ShopLandingPageApp', []);

    shopApp.directive('onLastRepeat', function() {
        return function(scope, element, attrs) {
            if (scope.$last) setTimeout(function(){
                scope.$emit('onRepeatLast', element, attrs);
            }, 1);
        };
    });

    function TrustpilotCtrl($scope, $http) {


        $scope.data = null;

        $scope.$on('onRepeatLast', function(scope, element, attrs){
            initializePagingCarousel('.cms-trustpilot-reviews');
        });

        $scope.reviewerVerbose = function(review) {
            var ret =  review.User.Name +", ";
            var d = new Date(review.Created.UnixTime * 1000);

            ret += d.getFullYear() + "-";
            ret += leadingZero(d.getMonth() + 1) + "-";
            ret += leadingZero(d.getDate()) + " ";
            ret += leadingZero(d.getHours()) + ":";
            ret += leadingZero(d.getMinutes()) + ":";
            ret += leadingZero(d.getSeconds());
            return ret ;
        };

        $http.jsonp('https://ssl.trustpilot.com/tpelements/374659/f.jsonp');
        window.trustpilot_jsonp_callback = function(data) {
            $scope.data = data;
        };

    }
    
    function leadingZero(convert) {
        return (convert < 10) ? ("0" + convert) : convert;
    }

    shopApp.controller('TrustpilotCtrl', ['$scope', '$http', TrustpilotCtrl]);

}