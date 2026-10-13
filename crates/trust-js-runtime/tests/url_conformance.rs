//! Focused WHATWG URL/UTS #46 and application/x-www-form-urlencoded API coverage.

use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;

use trust_js_runtime::{Completion, ConsoleOut, Runtime};

#[derive(Clone, Default)]
struct Captured(Rc<RefCell<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Captured {
    fn lines(&self) -> Vec<String> {
        String::from_utf8(self.0.borrow().clone())
            .expect("UTF-8 console output")
            .lines()
            .map(str::to_string)
            .collect()
    }
}

#[test]
fn url_state_machine_idna_setters_and_form_encoding() {
    let mut runtime = Runtime::new();
    let output = Captured::default();
    runtime.engine().ctx().op_state().put(ConsoleOut {
        out: Box::new(output.clone()),
        err: Box::new(Captured::default()),
    });

    let source = r##"
      console.log("idna", new URL("https://faß.example/").href);
      console.log("hosts", new URL("http://0x7f.1/").href,
        new URL("http://[2001:0db8::1]/").hostname);
      console.log("special", new URL("https:example.org").href,
        new URL("https://example.org\\a").href);
      console.log("opaque", new URL("mailto:some one@example.org?q=hello world#fragment").href,
        new URL("mailto:x").origin);
      console.log("file", new URL("file:c|/demo/../x").href);

      const url = new URL("http://example.com/a");
      url.pathname = "next value";
      url.host = "bücher.example:443";
      const beforeBadPort = url.href;
      url.port = "70000";
      console.log("set", url.href, url.href === beforeBadPort);
      url.search = "?";
      url.hash = "#";
      console.log("empty", url.href, JSON.stringify(url.search), JSON.stringify(url.hash));
      url.username = "a b";
      url.password = "p@ss";
      url.protocol = "https:";
      console.log("credentials", url.href, url.username, url.password);

      let params = new URLSearchParams("bad=%FF&x=%2&space=hello+world&escape=!'()~*");
      console.log("form", JSON.stringify(params.get("bad")), params.get("x"),
        params.get("space"), params.toString());
      params = new URLSearchParams([new Set(["a", "1"]), ["a", "2"], ["b", "3"]]);
      console.log("iterable", params.toString(), params.has("a", "2"));
      params.delete("a", "1");
      console.log("delete", params.toString());
      const iterator = params.entries();
      console.log("live1", JSON.stringify(iterator.next().value));
      params.append("c", "4");
      console.log("live2", JSON.stringify(iterator.next().value), JSON.stringify(iterator.next().value));
      console.log("usv", new URLSearchParams([["\ud800", "\udc00"]]).toString());
      console.log("primitive", new URLSearchParams(null).toString(),
        new URLSearchParams(5).toString());
      try { URL.canParse(); } catch (error) { console.log("required", error.name); }
      const nodeUrl = require("url");
      console.log("node-idna", nodeUrl.domainToASCII("faß.example"),
        nodeUrl.domainToUnicode("xn--fa-hia.example"), nodeUrl.domainToASCII("exa mple"));
    "##;

    match runtime.eval(source).expect("URL conformance script parses") {
        Completion::Value(_) => {}
        Completion::Throw { name, message } => panic!("uncaught {name}: {message}"),
    }

    assert_eq!(
        output.lines(),
        [
            "idna https://xn--fa-hia.example/",
            "hosts http://127.0.0.1/ [2001:db8::1]",
            "special https://example.org/ https://example.org/a",
            "opaque mailto:some one@example.org?q=hello%20world#fragment null",
            "file file:///c:/x",
            "set http://xn--bcher-kva.example:443/next%20value true",
            "empty http://xn--bcher-kva.example:443/next%20value?# \"\" \"\"",
            "credentials https://a%20b:p%40ss@xn--bcher-kva.example/next%20value?# a%20b p%40ss",
            "form \"�\" %2 hello world bad=%EF%BF%BD&x=%252&space=hello+world&escape=%21%27%28%29%7E*",
            "iterable a=1&a=2&b=3 true",
            "delete a=2&b=3",
            "live1 [\"a\",\"2\"]",
            "live2 [\"b\",\"3\"] [\"c\",\"4\"]",
            "usv %EF%BF%BD=%EF%BF%BD",
            "primitive null= 5=",
            "required TypeError",
            "node-idna xn--fa-hia.example faß.example ",
        ]
    );
}
