use std::collections::{HashMap, HashSet};

use naga::{Binding, VectorSize};
use proc_macro2::TokenStream;

/// Returns a base Rust or `glam` type that corresponds to a TypeInner, if one exists.
fn rust_type(type_inner: &naga::TypeInner) -> Option<syn::Type> {
    match type_inner {
        naga::TypeInner::Scalar(naga::Scalar { kind, width }) => match (kind, width) {
            (naga::ScalarKind::Bool, 1) => Some(syn::parse_quote!(bool)),
            (naga::ScalarKind::Float, 4) => Some(syn::parse_quote!(f32)),
            (naga::ScalarKind::Float, 8) => Some(syn::parse_quote!(f64)),
            (naga::ScalarKind::Sint, 4) => Some(syn::parse_quote!(i32)),
            (naga::ScalarKind::Sint, 8) => Some(syn::parse_quote!(i64)),
            (naga::ScalarKind::Uint, 4) => Some(syn::parse_quote!(u32)),
            (naga::ScalarKind::Uint, 8) => Some(syn::parse_quote!(u64)),
            _ => None,
        },
        naga::TypeInner::Vector {
            size,
            scalar: naga::Scalar { kind, width },
        } => {
            if cfg!(feature = "glam") {
                match (size, kind, width) {
                    (naga::VectorSize::Bi, naga::ScalarKind::Bool, 1) => {
                        Some(syn::parse_quote!(glam::bool::BVec2))
                    }
                    (naga::VectorSize::Tri, naga::ScalarKind::Bool, 1) => {
                        Some(syn::parse_quote!(glam::bool::BVec3))
                    }
                    (naga::VectorSize::Quad, naga::ScalarKind::Bool, 1) => {
                        Some(syn::parse_quote!(glam::bool::BVec4))
                    }
                    (naga::VectorSize::Bi, naga::ScalarKind::Float, 4) => {
                        Some(syn::parse_quote!(glam::f32::Vec2))
                    }
                    (naga::VectorSize::Tri, naga::ScalarKind::Float, 4) => {
                        Some(syn::parse_quote!(glam::f32::Vec3))
                    }
                    (naga::VectorSize::Quad, naga::ScalarKind::Float, 4) => {
                        Some(syn::parse_quote!(glam::f32::Vec4))
                    }
                    (naga::VectorSize::Bi, naga::ScalarKind::Float, 8) => {
                        Some(syn::parse_quote!(glam::f64::DVec2))
                    }
                    (naga::VectorSize::Tri, naga::ScalarKind::Float, 8) => {
                        Some(syn::parse_quote!(glam::f64::DVec3))
                    }
                    (naga::VectorSize::Quad, naga::ScalarKind::Float, 8) => {
                        Some(syn::parse_quote!(glam::f64::DVec4))
                    }
                    (naga::VectorSize::Bi, naga::ScalarKind::Sint, 4) => {
                        Some(syn::parse_quote!(glam::i32::IVec2))
                    }
                    (naga::VectorSize::Tri, naga::ScalarKind::Sint, 4) => {
                        Some(syn::parse_quote!(glam::i32::IVec3))
                    }
                    (naga::VectorSize::Quad, naga::ScalarKind::Sint, 4) => {
                        Some(syn::parse_quote!(glam::i32::IVec4))
                    }
                    (naga::VectorSize::Bi, naga::ScalarKind::Sint, 8) => {
                        Some(syn::parse_quote!(glam::i64::I64Vec2))
                    }
                    (naga::VectorSize::Tri, naga::ScalarKind::Sint, 8) => {
                        Some(syn::parse_quote!(glam::i64::I64Vec3))
                    }
                    (naga::VectorSize::Quad, naga::ScalarKind::Sint, 8) => {
                        Some(syn::parse_quote!(glam::i64::I64Vec4))
                    }
                    (naga::VectorSize::Bi, naga::ScalarKind::Uint, 4) => {
                        Some(syn::parse_quote!(glam::u32::UVec2))
                    }
                    (naga::VectorSize::Tri, naga::ScalarKind::Uint, 4) => {
                        Some(syn::parse_quote!(glam::u32::UVec3))
                    }
                    (naga::VectorSize::Quad, naga::ScalarKind::Uint, 4) => {
                        Some(syn::parse_quote!(glam::u32::UVec4))
                    }
                    (naga::VectorSize::Bi, naga::ScalarKind::Uint, 8) => {
                        Some(syn::parse_quote!(glam::u64::U64Vec2))
                    }
                    (naga::VectorSize::Tri, naga::ScalarKind::Uint, 8) => {
                        Some(syn::parse_quote!(glam::u64::U64Vec3))
                    }
                    (naga::VectorSize::Quad, naga::ScalarKind::Uint, 8) => {
                        Some(syn::parse_quote!(glam::u64::U64Vec4))
                    }
                    _ => None,
                }
            } else {
                None
            }
        }
        naga::TypeInner::Matrix {
            columns,
            rows,
            scalar: naga::Scalar { kind, width },
        } => {
            if !(cfg!(feature = "glam")) {
                return None;
            }
            if columns != rows {
                return None;
            }
            match (kind, width) {
                (naga::ScalarKind::Float, 4) => match columns {
                    naga::VectorSize::Bi => Some(syn::parse_quote!(glam::f32::Mat2)),
                    naga::VectorSize::Tri => Some(syn::parse_quote!(glam::f32::Mat3)),
                    naga::VectorSize::Quad => Some(syn::parse_quote!(glam::f32::Mat4)),
                },
                (naga::ScalarKind::Float, 8) => match columns {
                    naga::VectorSize::Bi => Some(syn::parse_quote!(glam::f64::Mat2)),
                    naga::VectorSize::Tri => Some(syn::parse_quote!(glam::f64::Mat3)),
                    naga::VectorSize::Quad => Some(syn::parse_quote!(glam::f64::Mat4)),
                },
                _ => None,
            }
        }
        naga::TypeInner::Atomic(scalar) => rust_type(&naga::TypeInner::Scalar(*scalar)),
        _ => None,
    }
}

/// A builder for type definition and identifier pairs.
pub struct TypesDefinitions {
    definitions: Vec<syn::ItemStruct>,
    references: HashMap<naga::Handle<naga::Type>, syn::Type>,
    structs_filter: Option<HashSet<String>>,
}

impl TypesDefinitions {
    /// Constructs a new type definition collator, with a given filter for type names.
    pub fn new(module: &naga::Module, structs_filter: Option<HashSet<String>>) -> Self {
        let mut res = Self {
            definitions: Vec::new(),
            references: HashMap::new(),
            structs_filter,
        };

        for (ty_handle, _) in module.types.iter() {
            if let Some(new_ty_ident) = res.try_make_type(ty_handle, module, true) {
                res.references.insert(ty_handle, new_ty_ident.clone());
            }
        }

        return res;
    }

    fn try_make_type(
        &mut self,
        ty_handle: naga::Handle<naga::Type>,
        module: &naga::Module,
        omit_underscore_prefixed: bool,
    ) -> Option<syn::Type> {
        let ty = match module.types.get_handle(ty_handle) {
            Err(_) => return None,
            Ok(ty) => ty,
        };
        if let Some(ty_ident) = rust_type(&ty.inner) {
            return Some(ty_ident);
        };

        match &ty.inner {
            naga::TypeInner::Array { base, size, .. }
            | naga::TypeInner::BindingArray { base, size } => {
                let base_type = self.rust_type_ident(*base, module)?;
                match size {
                    naga::ArraySize::Constant(size) => {
                        let size = size.get();
                        Some(syn::parse_quote!([#base_type; #size as usize]))
                    }
                    naga::ArraySize::Dynamic => Some(syn::parse_quote!(Vec<#base_type>)),
                }
            }
            naga::TypeInner::Struct { members, .. } => {
                let struct_name = ty.name.as_deref();
                let struct_name = match struct_name {
                    None => return None,
                    Some(struct_name) => struct_name,
                };

                // Apply filter
                if let Some(struct_name_filter) = &self.structs_filter {
                    if !struct_name_filter.contains(struct_name) {
                        return None;
                    }
                }

                let struct_name = syn::parse_str::<syn::Ident>(struct_name).ok();

                if struct_name.is_none() {
                    return None;
                }

                let members_have_names = members.iter().all(|member| member.name.is_some());
                let members: Vec<_> = members
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i_member, member)| {
                        let member_name = if members_have_names {
                            let member_name =
                                member.name.as_ref().expect("all members had names").clone();
                            syn::parse_str::<syn::Ident>(&member_name)
                        } else {
                            syn::parse_str::<syn::Ident>(&format!("v{}", i_member))
                        };
                        let member_ty = self.rust_type_ident(member.ty, module);
                        let member_wgpu_ty = self.wgpu_type_ident(member.ty, module);

                        let mut attributes = proc_macro2::TokenStream::new();
                        // Runtime-sized fields must be marked as such when using encase
                        if cfg!(feature = "encase") {
                            let ty = module.types.get_handle(member.ty);
                            if let Ok(naga::Type { inner, .. }) = ty {
                                match inner {
                                    naga::TypeInner::Array {
                                        size: naga::ArraySize::Dynamic,
                                        ..
                                    }
                                    | naga::TypeInner::BindingArray {
                                        size: naga::ArraySize::Dynamic,
                                        ..
                                    } => attributes.extend(quote::quote!(#[size(runtime)])),
                                    _ => {}
                                }
                            }
                        }

                        let member_name = match member_name {
                            Ok(member_name_ident) => member_name_ident,
                            Err(_) => return None,
                        };

                        let member_ty = match member_ty {
                            Some(member_ty) => member_ty,
                            None => return None,
                        };

                        let token_stream = if member_name.to_string().starts_with('_')
                            && omit_underscore_prefixed
                        {
                            quote::quote! {
                                #attributes
                                #member_name: #member_ty
                            }
                        } else {
                            quote::quote! {
                                #attributes
                                pub #member_name: #member_ty
                            }
                        };

                        let binding = match (&member.binding, member_wgpu_ty) {
                            (Some(binding), Some(wgpu_ty)) => Some((binding, wgpu_ty)),
                            _ => None,
                        };

                        Some((member_name, token_stream, binding))
                    })
                    .collect();

                let bindings = members.iter().filter(|(_, _, binding)| binding.is_some());
                let location_bindings =
                    members
                        .iter()
                        .filter_map(|(member_name, _, binding)| match binding {
                            Some((Binding::Location { location, .. }, wgpu_ty)) => {
                                Some((member_name, wgpu_ty, location))
                            }
                            _ => None,
                        });

                #[allow(unused_mut)]
                let mut bonus_struct_derives = TokenStream::new();

                let mut push_struct_def = || {
                    let member_token_streams =
                        members.iter().map(|(_, token_stream, _)| token_stream);

                    self.definitions.push(syn::parse_quote! {
                        #[allow(unused, non_camel_case_types)]
                        #[derive(Debug, PartialEq, Clone, Default, #bonus_struct_derives)]
                        pub struct #struct_name {
                            #(#member_token_streams ,)*
                        }
                    });
                };

                if !bindings.clone().next().is_some() {
                    // this is a uniform
                    #[cfg(feature = "encase")]
                    bonus_struct_derives.extend(quote::quote!(encase::ShaderType,));

                    push_struct_def();
                }

                if location_bindings.clone().next().is_some() {
                    // this is a vertex buffer. generate the wgpu::VertexBufferLayout, and don't derive encase::ShaderType
                    #[cfg(feature = "bytemuck")]
                    bonus_struct_derives.extend(quote::quote!(bytemuck::Pod, bytemuck::Zeroable,));

                    push_struct_def();

                    #[cfg(feature = "wgsl")]
                    {
                        let location_count = location_bindings.clone().count();

                        let attributes =
                            location_bindings.map(|(member_name, member_wgpu_ty, location)| {
                                quote::quote! {
                                    wgpu::VertexAttribute {
                                        format: #member_wgpu_ty,
                                        offset: offset_of!(#struct_name, #member_name) as _,
                                        shader_location: #location,
                                    },
                                }
                            });

                        self.definitions.push(syn::parse_quote! {
                            impl #struct_name {
                                pub const fn desc(step_mode: wgpu::VertexStepMode::Vertex) -> wgpu::VertexBufferLayout<'static> {
                                    const ATTRIBUTES: [wgpu::VertexAttribute; #location_count] = [
                                        #(#attributes )*
                                    ];

                                    wgpu::VertexBufferLayout {
                                        array_stride: core::mem::size_of::<Self>() as wgpu::BufferAddress,
                                        step_mode,
                                        attributes: &ATTRIBUTES,
                                    }
                                }
                            }
                        });
                    }
                }

                Some(syn::parse_quote!(#struct_name))
            }
            _ => None,
        }
    }

    /// Takes a handle to a type, and a module where the type resides, and tries to return an identifier
    /// of that type, in Rust. Note that for structs this will be an identifier in to the set of structs generated
    /// by calling `TypesDefinitions::definitions()`, so your output should make sure to include everything from
    /// there in the scope where the returned identifier is used.
    pub fn rust_type_ident(
        &mut self,
        ty_handle: naga::Handle<naga::Type>,
        module: &naga::Module,
    ) -> Option<syn::Type> {
        if let Some(ident) = self.references.get(&ty_handle).cloned() {
            return Some(ident);
        }

        if let Some(built) = self.try_make_type(ty_handle, module, true) {
            self.references.insert(ty_handle, built.clone());
            return Some(built);
        }

        return None;
    }

    /// Takes a handle to a type, and a module where the type resides, and tries to return an identifier
    /// of that type, as a wgpu::VertexFormat.
    pub fn wgpu_type_ident(
        &mut self,
        ty_handle: naga::Handle<naga::Type>,
        module: &naga::Module,
    ) -> Option<syn::Type> {
        let ty = module.types.get_handle(ty_handle).ok()?;

        match &ty.inner {
            #[rustfmt::skip]
            naga::TypeInner::Scalar(scalar) => {
                match (scalar.kind, scalar.width) {
                    (naga::ScalarKind::Float, 4) => Some(syn::parse_quote!(wgpu::VertexFormat::Float32)),
                    (naga::ScalarKind::Uint, 4) => Some(syn::parse_quote!(wgpu::VertexFormat::Uint32)),
                    (naga::ScalarKind::Sint, 4) => Some(syn::parse_quote!(wgpu::VertexFormat::Sint32)),
                    (naga::ScalarKind::Float, 8) => Some(syn::parse_quote!(wgpu::VertexFormat::Float64)),
                    _ => None,
                }
            },
            #[rustfmt::skip]
            naga::TypeInner::Vector { size, scalar } => {
                match (scalar.kind, scalar.width, size) {
                    (naga::ScalarKind::Uint, 1, VectorSize::Bi) => Some(syn::parse_quote!(wgpu::VertexFormat::Uint8x2)),
                    (naga::ScalarKind::Uint, 1, VectorSize::Quad) => Some(syn::parse_quote!(wgpu::VertexFormat::Uint8x4)),
                    (naga::ScalarKind::Sint, 1, VectorSize::Bi) => Some(syn::parse_quote!(wgpu::VertexFormat::Sint8x2)),
                    (naga::ScalarKind::Sint, 1, VectorSize::Quad) => Some(syn::parse_quote!(wgpu::VertexFormat::Sint8x4)),
                    (naga::ScalarKind::Uint, 2, VectorSize::Bi) => Some(syn::parse_quote!(wgpu::VertexFormat::Uint16x2)),
                    (naga::ScalarKind::Uint, 2, VectorSize::Quad) => Some(syn::parse_quote!(wgpu::VertexFormat::Uint16x4)),
                    (naga::ScalarKind::Sint, 2, VectorSize::Bi) => Some(syn::parse_quote!(wgpu::VertexFormat::Sint16x2)),
                    (naga::ScalarKind::Sint, 2, VectorSize::Quad) => Some(syn::parse_quote!(wgpu::VertexFormat::Sint16x4)),
                    (naga::ScalarKind::Float, 2, VectorSize::Bi) => Some(syn::parse_quote!(wgpu::VertexFormat::Float16x2)),
                    (naga::ScalarKind::Float, 2, VectorSize::Quad) => Some(syn::parse_quote!(wgpu::VertexFormat::Float16x4)),
                    (naga::ScalarKind::Uint, 4, VectorSize::Bi) => Some(syn::parse_quote!(wgpu::VertexFormat::Uint32x2)),
                    (naga::ScalarKind::Uint, 4, VectorSize::Tri) => Some(syn::parse_quote!(wgpu::VertexFormat::Uint32x3)),
                    (naga::ScalarKind::Uint, 4, VectorSize::Quad) => Some(syn::parse_quote!(wgpu::VertexFormat::Uint32x4)),
                    (naga::ScalarKind::Sint, 4, VectorSize::Bi) => Some(syn::parse_quote!(wgpu::VertexFormat::Sint32x2)),
                    (naga::ScalarKind::Sint, 4, VectorSize::Tri) => Some(syn::parse_quote!(wgpu::VertexFormat::Sint32x3)),
                    (naga::ScalarKind::Sint, 4, VectorSize::Quad) => Some(syn::parse_quote!(wgpu::VertexFormat::Sint32x4)),
                    (naga::ScalarKind::Float, 4, VectorSize::Bi) => Some(syn::parse_quote!(wgpu::VertexFormat::Float32x2)),
                    (naga::ScalarKind::Float, 4, VectorSize::Tri) => Some(syn::parse_quote!(wgpu::VertexFormat::Float32x3)),
                    (naga::ScalarKind::Float, 4, VectorSize::Quad) => Some(syn::parse_quote!(wgpu::VertexFormat::Float32x4)),
                    _ => None,
                }
            },
            _ => None,
        }
    }

    /// Gives the set of definitions required by the identifiers generated by this object. These should be
    /// emitted somewhere accessable by the places that the identifiers were used.
    pub fn definitions(self) -> Vec<syn::Item> {
        self.definitions
            .into_iter()
            .map(|item_struct| syn::Item::Struct(item_struct))
            .collect()
    }
}
