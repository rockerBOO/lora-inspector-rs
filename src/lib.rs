use web_sys::{
    js_sys, Event, FileReader, HtmlInputElement, MessageEvent, ProgressEvent, SharedWorker,
};

use crate::worker::LoraWorker;

mod inspector;
mod metadata;
mod norms;
mod worker;

#[derive(Debug)]
pub enum Error {
    Candle(candle_core::Error),
    SafeTensor(safetensors::SafeTensorError),
    Load(String)
}

// fn main() {
//     use std::cell::RefCell;
//     use std::rc::Rc;
//     use wasm_bindgen::prelude::*;
//     use web_sys::console;
//
//     #[wasm_bindgen]
//     struct WorkerMessage {
//         message_type: String,
//         payload: JsValue,
//     }
//
//     /// Run entry point for the main thread.
//     #[wasm_bindgen]
//     pub fn startup() {
//         // Here, we create our worker. In a larger app, multiple callbacks should be
//         // able to interact with the code in the worker. Therefore, we wrap it in
//         // `Rc<RefCell>` following the interior mutability pattern. Here, it would
//         // not be needed but we include the wrapping anyway as example.
//         let worker_handle = Rc::new(RefCell::new(SharedWorker::new("./worker.js").unwrap()));
//         // worker_handle
//         //     .borrow()
//         //     .port()
//         //     .post_message(&text_message("register", "hello from WASM"));
//         console::log_1(&"Created a new worker from within Wasm".into());
//
//         // Pass the worker to the function which sets up the `oninput` callback.
//         setup_input_oninput_callback(worker_handle);
//     }
//
//     fn setup_input_oninput_callback(worker: Rc<RefCell<web_sys::SharedWorker>>) {
//         console_error_panic_hook::set_once();
//         let document = web_sys::window().unwrap().document().unwrap();
//
//         // If our `onmessage` callback should stay valid after exiting from the
//         // `oninput` closure scope, we need to either forget it (so it is not
//         // destroyed) or store it somewhere. To avoid leaking memory every time we
//         // want to receive a response from the worker, we move a handle into the
//         // `oninput` closure to which we will always attach the last `onmessage`
//         // callback. The initial value will not be used and we silence the warning.
//         #[allow(unused_assignments)]
//         let mut persistent_callback_handle = get_on_msg_callback(Rc::clone(&worker));
//         console::log_1(&"setup on input message".into());
//
//         let callback: Closure<dyn FnMut(ProgressEvent)> =
//             Closure::new(move |event: ProgressEvent| {
//                 console::log_1(&"oninput callback triggered".into());
//                 let document = web_sys::window().unwrap().document().unwrap();
//                 let input_field = document
//                     .get_element_by_id("inputNumber")
//                     .expect("#inputNumber should exist");
//                 let input_field = input_field
//                     .dyn_ref::<HtmlInputElement>()
//                     .expect("#inputNumber should be a HtmlInputElement");
//                 let files = input_field.files().expect("to get files");
//                 let file = files.item(0).expect("to get a file");
//
//                 let file_name = file.name();
//                 console::log_1(&format!("got file upload. {file_name}").into());
//
//                 // let worker_handle = &*worker.borrow();
//
//                 let worker_handle = Rc::clone(&worker);
//
//                 let mut onload = Closure::wrap(Box::new(move |event: Event| {
//                     let file_reader: FileReader = event.target().unwrap().dyn_into().unwrap();
//                     let file = file_reader.result().unwrap();
//                     let file = js_sys::Uint8Array::new(&file);
//
//                     // let mut worker_file_handle2 = worker_file_handle.clone();
//
//                     let file_length = file.length();
//                     console::log_1(&format!("got file upload. {file_length}").into());
//                     let mut worker_f = worker_file_handle.borrow_mut();
//                     // worker_f.set_buffer(file.to_vec());
//                     //
//                     // let len = &worker_f.buffer.as_ref().unwrap().len();
//                     // console::log_1(&format!("worker_file_handle {len}").into());
//
//                     // let worker_file = WorkerFile::new(&file.to_vec().to_owned());
//
//                     // worker_handle
//                     //     .borrow()
//                     //     .port()
//                     //     .post_message(&text_message("upload", &file_name));
//
//                     // let mut psd_file = vec![0; psd.length() as usize];
//                     // file.copy_to(&mut psd_file);
//
//                     // store.borrow_mut().msg(&Msg::ReplacePsd(&psd_file));
//                 }) as Box<dyn FnMut(Event)>);
//
//                 let file_reader = web_sys::FileReader::new().unwrap();
//                 file_reader.read_as_array_buffer(&file).unwrap();
//                 file_reader.set_onload(Some(onload.as_ref().unchecked_ref()));
//
//                 // let msg_worker_file = Rc::clone(&worker_file_handle);
//                 // let worker_handle = Rc::clone(&worker_handle);
//                 persistent_callback_handle = get_on_msg_callback(Rc::clone(&worker));
//                 worker
//                     .borrow()
//                     .port()
//                     .set_onmessage(Some(persistent_callback_handle.as_ref().unchecked_ref()));
//                 onload.forget();
//             });
//
//         // Attach the closure as `oninput` callback to the input field.
//         document
//             .get_element_by_id("inputNumber")
//             .expect("#inputNumber should exist")
//             .dyn_ref::<HtmlInputElement>()
//             .expect("#inputNumber should be a HtmlInputElement")
//             .set_oninput(Some(callback.as_ref().unchecked_ref()));
//
//         // Leaks memory.
//         callback.forget();
//     }
//
//     /// Create a closure to act on the message returned by the worker
//     fn get_on_msg_callback(
//         worker: Rc<RefCell<web_sys::SharedWorker>>,
//     ) -> Closure<dyn FnMut(MessageEvent)> {
//         Closure::new(move |event: MessageEvent| {
//             console::log_1(&format!("WAMS ggot a message! ").into());
//         })
//     }
// }
