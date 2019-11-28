/**
* hoverIntent r6 // 2011.02.26 // jQuery 1.5.1+
* <http://cherne.net/brian/resources/jquery.hoverIntent.html>
* 
* @param  f  onMouseOver function || An object with configuration options
* @param  g  onMouseOut function  || Nothing (use configuration options object)
* @author    Brian Cherne brian(at)cherne(dot)net
*/
(function($){$.fn.hoverIntent=function(f,g){var cfg={sensitivity:7,interval:100,timeout:0};cfg=$.extend(cfg,g?{over:f,out:g}:f);var cX,cY,pX,pY;var track=function(ev){cX=ev.pageX;cY=ev.pageY};var compare=function(ev,ob){ob.hoverIntent_t=clearTimeout(ob.hoverIntent_t);if((Math.abs(pX-cX)+Math.abs(pY-cY))<cfg.sensitivity){$(ob).unbind("mousemove",track);ob.hoverIntent_s=1;return cfg.over.apply(ob,[ev])}else{pX=cX;pY=cY;ob.hoverIntent_t=setTimeout(function(){compare(ev,ob)},cfg.interval)}};var delay=function(ev,ob){ob.hoverIntent_t=clearTimeout(ob.hoverIntent_t);ob.hoverIntent_s=0;return cfg.out.apply(ob,[ev])};var handleHover=function(e){var ev=jQuery.extend({},e);var ob=this;if(ob.hoverIntent_t){ob.hoverIntent_t=clearTimeout(ob.hoverIntent_t)}if(e.type=="mouseenter"){pX=ev.pageX;pY=ev.pageY;$(ob).bind("mousemove",track);if(ob.hoverIntent_s!=1){ob.hoverIntent_t=setTimeout(function(){compare(ev,ob)},cfg.interval)}}else{$(ob).unbind("mousemove",track);if(ob.hoverIntent_s==1){ob.hoverIntent_t=setTimeout(function(){delay(ev,ob)},cfg.timeout)}}};return this.bind('mouseenter',handleHover).bind('mouseleave',handleHover)}})(jQuery);/**
 * All functions for the header
 *
 * Dependencies: jquery-1.3.2.min.js,jquery-ui-1.7.2.core.min.js,jquery-ui-1.7.2.effects.min.js
 */
jQuery(function($){

	/** MANIPULATE DOM **/
	positioningSubNavigation();

	/** header navigation **/
	showSubnavigationLayers();
	registerMainNavigationResponsiveClick();
	
	addReturnUrlForLinks();
	
	/** initialize search bar **/
	setSearchQueryTextFromUrlParameter();

	/** move messages for top of the page as first children of body */
	moveBannerMessageToTop();

	/* Menu on dropdown buttons
	------------------------------------------------------------------------------*/
	$('.pillow-btn, .pillow-btn-header')
		.not('.pillow-btn-processed')
		.click(function() {
			$(this).toggleClass('pillow-btn-active');
		})
		.addClass('pillow-btn-processed');
	$('.open-menu').click(function() {
		$('body').toggleClass('show-menu');
	});
	$('.open-search').click(function() {
		$('body').toggleClass('show-search');
	});
	$('.open-legal')
		.not('.open-legal-processed')
		.click(function(){
			$("body").toggleClass("show-legal")
		}).addClass('open-legal-processed');
	
    /* Flyout
	------------------------------------------------------------------------------*/
	var doNotOpenOnMouseOver = {
      over: function(){} , // function = onMouseOver callback (REQUIRED)
      timeout: 500, // number = milliseconds delay before onMouseOut
      out: function(){
    	  if ($('input:focus', this).length <= 0) {
              $(this).removeClass('is-open').find('.pillow-btn-active').removeClass('pillow-btn-active');
              $('body').removeClass('show-footer-menu');
          }
      }
    };
	// open flyouts on click
    $('.flyout').click(function(e){
          // force flyout to open/close also on click (for touch devices)
          if($(this).hasClass('is-open')){
              if(!$('form',this).length){
                $(this).removeClass('is-open');
              }
          }
          else{
            $(this).addClass('is-open');
        }
        e.stopPropagation(); // don't let event bubble up to document (for closing by clicking outside)
     });
    $("#auth").hoverIntent(doNotOpenOnMouseOver);
    $("#lang").hoverIntent(doNotOpenOnMouseOver);
    

     $(document).click(function(){
       $('.flyout.is-open').removeClass('is-open').find('.pillow-btn-active').removeClass('pillow-btn-active');
     });
     // close flyout on click outside

    function addReturnUrlForLinks() {
    	$('a.returnURL').each(function(){
    		if (!this.search.match(/[&?]returnURL=/)) {
    			this.search += (this.search ? '&' : '?') + 'returnURL=' + encodeURIComponent(window.location.href); 
    		}
    	});
    }
    
	$('.logoutButton').click(
		function(e) {
			e.preventDefault();			
			var domain = window.location.hostname;
			var simUserCookie = $.cookie('sim-user-token');
			if (simUserCookie === 'undefined') {
				return;
			}
			var domainParts = window.location.hostname.split(".");
			for (var startIndex=0; startIndex<domainParts.length; startIndex++) {
				var domain = "";
				for (var i=startIndex; i<domainParts.length; i++) {
					domain += "." + domainParts[i]; 
				}
				removeUserTokenCookies(domain);				
				removeUserTokenCookies(domain.replace(/^[.]/,''));
			}
			if(window.location.search.indexOf('IFormSubmitListener') > 0) {
				// do not resubmit forms on logout
				window.location = window.location.href.split('?')[0];
			} else {
				logoutRequest();
			}
		}
	);
	
	function logoutRequest(){
		try {
			if (!window.location.origin) {
				window.location.origin = window.location.protocol + "//" + window.location.hostname + (window.location.port ? ':' + window.location.port: '');
			}
			return $.get(window.location.origin + "/spcom-resources/logout/",function(data) { window.location.reload(); });
		} catch(err) {
			// nothing to worry about
		}
	}

	function registerMainNavigationResponsiveClick() {
		$('.mainNavigationLinkMark').on('click', function(e) {
			var $li = $(this).closest('.mainNavigationMark');
			if($li.find('.nav-sub').length > 0) {
				e.preventDefault();
				var $lis = $li.closest('#global-nav').children();
				if($li.hasClass('open-clicked')) {
					$li.removeClass('open-clicked');
				} else {
					$li.addClass('open-clicked');
				}
				return false;
			}
		});
	}

	function showSubnavigationLayers(){
		
		var mainNavConfig = {
			timeout: 300, // number = milliseconds delay before onMouseOver / onMouseOut
			over: function() {
				$('.colLeft.zIndex').removeClass('zIndex');
				$('.colRight.zIndex').removeClass('zIndex');
				$(this).addClass('open');
			},
			out: function() {
				$(this)
					.removeClass('open');
			}
		};

		$('nav li.mainNavigationMark').hoverIntent(mainNavConfig);
		
		// close nav layers on click outside of layers; needed for devices without onmouseout (tablets)
		$('body').on('click', function() {
			$('nav li.mainNavigationMark').removeClass('open').removeClass('open-clicked');
		});
	}

	function positioningSubNavigation() {
		// position the main navigation layer
		var navWidth = $("nav").width();
		var layerMarginRight = 10;
		if($('.lt-ie8').length == 0) {
			$("nav > ul > li").each(function() {
				var $this = $(this);
				var $layer = $('.nav-sub', $this);
				var offsetLabel = $this.position().left;
				var widthLabel = $this.width();
				var widthLayer = $layer.width();
				if(offsetLabel + widthLayer > navWidth) {
					$layer.css('left', 'auto');
					$layer.css('right', '-' + (navWidth - offsetLabel - widthLabel - layerMarginRight) + 'px');
				}
			});
		}
	}
	
	function setSearchQueryTextFromUrlParameter() {
		var queryMatch = /[?&]queryText=([^&?\s]+)/.exec(window.location.search);
		if (queryMatch) {
			var $inputField = $('input#query');
			$inputField.val($inputField.val() || decodeURIComponent(queryMatch[1].replace(/\+/g,' ')));
		}
	}

	function moveBannerMessageToTop() {
		$('.banner-message').detach().prependTo('body');
	}

	svgFallback();

});

function reloadHeader(callback) {
	try{
		$('header').load(setParamValue($('header').data('src'), 'sim-user-token', $.cookie('sim-user-token')), callback);
	}catch(err) {
		// nothing to worry about
	}
}

function removeUserTokenCookies(domain){
	$.removeCookie('sim-user-token',{ domain : domain, path: '/' });
	$.removeCookie('SPCOMSESSIONID',{ domain : domain, path: '/' });
	$.removeCookie('UID',{ domain : domain, path: '/' });
	$.removeCookie('spcomLoggedIn',{ domain : domain, path: '/' });
	$.removeCookie('beechwood_authentication',{ domain : domain, path: '/' });
}
function setParamValue(url, key, value){
	key = encodeURIComponent(key);
	value = encodeURIComponent(value || "");
	var regexp = new RegExp('([&?]'+key+'=)[^&]*');
	if (url.match(regexp)) {
		return url.replace(regexp, '$1'+value);
	} else {
		return url + (url.indexOf('?') > 0 ? '&' : '?') + key + '=' + value;
	}
}

/**
 * Check for currently clicking on a a-href-link.
 * Used by {@link com.artnology.sgw.cda.general.behaviors.ObfuscatedLinkBehavior}
 * <p>
 * see e.g. http://stackoverflow.com/questions/4762254/javascript-window-location-does-not-set-referer-in-the-request-header
 * @param url URL to go to
 * @param anchor HTML link element, if the user currently clicks on
 * @return true if clicked on anchor (thus anchor will handle clicking itself),
 * 	       false otherwise
 */
function goURL(url, anchor) {
	if (anchor && (anchor.tagName === 'A')) {
		// if clicking on a link then really click on that link to have the browser's referrer set correctly
		anchor.href = url;
		return true;
	} else {
		// in any other case just set window.location.href
		window.location.href = url;
		return false;
	}
}

svgFallback = function() {
	
	function svgasimg() {
		return document.implementation.hasFeature('http://www.w3.org/TR/SVG11/feature#Image', '1.1');
	}
	
	if (!svgasimg()){
		var e = document.getElementsByTagName('img');
		if (!e.length){
			e = document.getElementsByTagName('IMG');
		}
		for (var i=0, n=e.length; i<n; i++){
			var img = e[i],
			src = img.getAttribute("src");
			if (src.match(/svg$/)) {
				if (img.hasAttribute('data-fallback')) {
					img.setAttribute('src', img.getAttribute('data-fallback'));
				} else {
					img.setAttribute('src', src.replace(/svg$/, 'png'));
				}
			}
		}
	}
}

// CART WIDGET
$(function(){
	var rootEle = '#cross-nav #basket-widget ';
	
	function updateCartWidget(){
		var basketId = getCookie('SPRCOMBASKET');
		var simUserId = getCookie('sim-user-token');
		if (isCookieValueDefined(basketId)) {
			updateBasket(basketId, simUserId);
		} else if (isCookieValueDefined(simUserId)) {
			updateBasket("notdefined", simUserId);
		}
	}
	function updateBasket(basketId, simUserId){
		$(rootEle).addClass("loading");
        getBasket(basketId, simUserId).done(function(basketData){
            render(basketData);
        }).fail(function(response){
            console.log("updateCartWidget failed");
            console.log(response.status);
            console.log(response.responseText);
            $(rootEle).removeClass("loading");
            $(rootEle).addClass("empty");
        });
	}
	function getBasket(basketId, simUserId){
		var host = window.location.protocol+"//" + window.location.hostname;
		var query = '';
		if (isCookieValueDefined(simUserId)) {
			query += '?sim-token=' + simUserId;
		}
		
		var springercomcountry = getCookie('springercomcountry');
		var siteCountry = 'ES'; // default
		if (isCookieValueDefined(springercomcountry)) {
			siteCountry = springercomcountry;
		}
		
		return $.get(host + "/api/basket/" + siteCountry + "/" + basketId + query);
	}
	function isCookieValueDefined(value) {
		return (value !== 'undefined' && value.length > 0)
	}
	function getSitePrefix(){
		return $(rootEle).attr("data-site-prefix");
	}
	function getLinkToCartPage(){
		return $(rootEle).attr("data-cart-page-link");
	}
	function render(basketData){
		$(rootEle).removeClass("loading");
		$(rootEle +"ul").html(renderItems(basketData));
		$(rootEle + ".totalQuantity").html(basketData.totalQuantity + " " + basketData.resources.items);
		$(rootEle + ".cart-summary .shipping .left-column").html(basketData.resources.shipping);
		$(rootEle + ".cart-summary .shipping .right-column").html(basketData.subtotal);
		$(rootEle + ".cart-summary .total .left-column").html(basketData.resources.subtotal);
		$(rootEle + ".cart-summary .total .right-column").html(basketData.totalPrice);
		$(rootEle + ".cart-summary .view-cart").attr("href", getLinkToCartPage());
		$(rootEle + ".cart-summary .view-cart").html(basketData.resources.viewCart);
		if(basketData.totalQuantity == 0){
			$(rootEle).addClass("empty");
		} else {
			$(rootEle).removeClass("empty");
			if(basketData.items.length > 5){
				$(rootEle).addClass("full");
			} else {
				$(rootEle).removeClass("full");
			}
		}
	}
	
	var itemTemplate = $('#cart-widget-cart-item').html();
	function renderItems(basketData){
		return basketData.items.slice(0,5).map(function(item){
			var data2 = {
					resources : basketData.resources,
					coverTemplate : renderCover(item, basketData.resources),
					titleTemplate : renderTitle(item)
			}
			return renderTemplate(itemTemplate, item, data2);
		}).join(" "); 
	}
	var coverWithLinkTemplate = $('#cart-widget-cover-with-link').html();
	var coverWithoutLinkTemplate = $('#cart-widget-cover-without-link').html();
	function renderCover(basketItem, data2){
		if(basketItem.productLinkFragment){
			return renderTemplate(coverWithLinkTemplate, basketItem, data2);
		} else {
			return renderTemplate(coverWithoutLinkTemplate, basketItem, data2);
		}
	}
	function renderTitle(basketItem){
		if(basketItem.productLinkFragment){
			return renderTemplate('<a href="{{productLinkFragment}}">{{title}}</a>', basketItem);
		} else {
			return renderTemplate('{{title}}', basketItem);
		}
	}
	// this could be replaced with a proper templating library in the future
	// allows to pass 2 data objects
	// in the template you can use expressions like {{attribute}} or {{attribute.subAttribute}}
	function renderTemplate(s,data1, data2){ 
		var rawExpressions = s.match(new RegExp('{{.*?}}','g')); // yields an array of "{{attribute}}"
		var expressions = map(rawExpressions, function(expr){ // then remove curly braces to get just the names
			return expr.replace(new RegExp('{{','g'),'').replace(new RegExp('}}','g'),'');
		});
		forEach(expressions, function(expression){
			 var replacement = getDescendantProp(data1, expression);
			 if(replacement == null && data2){				 
				 replacement = getDescendantProp(data2, expression);
			 }
			 if(replacement != null){				 
				 s = s.replace(new RegExp('{{'+ expression +'}}','g'), replacement);
			 }
		});
		s = s.replace(new RegExp('data-src','g'), 'src').replace(new RegExp('data-href','g'), 'href');
		s = s.replace(new RegExp('../../../','g'), ''); // this stuff is inserted by the browser into empty src attributes
		return s;
	}
	function forEach(arr, fn){
		for (var i = 0; i < arr.length; i++) {
			fn(arr[i]);
		}
	}
	function map(arr, fn){
		var ret = [];
		for (var i = 0; i < arr.length; i++) {
			ret.push(fn(arr[i]));
		}
		return ret;
	}
	function getDescendantProp(obj, desc) { // http://stackoverflow.com/questions/8051975/access-object-child-properties-using-a-dot-notation-string
	    var arr = desc.split(".");
	    while(arr.length && (obj = obj[arr.shift()]));
	    return obj;
	}
	function getCookie(cname) { // we do not want to rely on a library - taken from: http://www.w3schools.com/js/js_cookies.asp
	    var name = cname + "=";
	    var cookieArray = document.cookie.split(';');
	    for(var i=0; i<cookieArray.length; i++) {
	        var c = cookieArray[i];
	        while (c.charAt(0)==' ') c = c.substring(1);
	        if (c.indexOf(name) == 0) return c.substring(name.length,c.length);
	    }
	    return "";
	}
	
	(function(fn){// lte ie8 have no array.map
		if (!fn.map) fn.map=function(f){var r=[];for(var i=0;i<this.length;i++)r.push(f(this[i]));return r}
		if (!fn.filter) fn.filter=function(f){var r=[];for(var i=0;i<this.length;i++)if(f(this[i]))r.push(this[i]);return r}
	})(Array.prototype);
	
	function isDesktop(){
		if(window.matchMedia){			
			return window.matchMedia("(min-width: 1004px)").matches; // taken from springer-compass file -> _respond-to.scss
		} else {
			return true; // assume desktop size if matchMedia function is not available
		}
	}
	function isCda(){ 
		return $(".wrapper").hasClass("cda");
	}
	function isMobileView(){
		return !isDesktop() && !isCda();
	}
	// render the contents initially
	updateCartWidget();
	// register for update events
	$(document).on("basket-updated", updateCartWidget);
	
	// implement correct click behavior on mobile
	$("#basket-widget.flyout").click(function(e){
		if(isMobileView()){			
			$(this).removeClass('is-open');// remove this class, which was added by the original flyout click handler
			window.location = getLinkToCartPage();
		}
	});
	
	// open on hover	
	$("#basket-widget").hoverIntent({
      over: function(){
    	  if(!isMobileView()){    		  
    		  $(this).addClass('is-open'); 
    	  }
      },
      timeout: 500, // number = milliseconds delay before onMouseOut
      out: function(){
    	  $(this).removeClass('is-open');
    }});
	// export API
	window.basketWidget = {
		refresh : updateCartWidget
	}
});