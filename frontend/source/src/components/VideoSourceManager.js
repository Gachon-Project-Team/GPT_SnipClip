import React, { useState } from 'react';
import SectionFormatter from './SectionFormatter';
import { ManagerContainer, SubmitButton, SubmitButtonWrapper } from './Styles/Container/VideoSourceManager.styles';

function VideoSourceManager({ realResult, aiResult, onSubmit }) {
    const [sections, setSections] = useState(() => {
        const combined = [];
        for (let i = 0; i < 5; i++) {
            const text = `${realResult.section[i * 2]} ${realResult.section[i * 2 + 1]}`;
            combined.push({
                script: text,
                realImage: realResult.image[i],
                aiImage: aiResult.image[i],
                selectedImageType: 'real',
            });
        }
        return combined;
    });
    const handleScriptChange = (index, newScript) => {
        setSections(prev =>
            prev.map((item, i) =>
                i === index ? { ...item, script: newScript } : item
            )
        );
    };
    const handleImageTypeChange = (index, newType) => {
        setSections(prev =>
            prev.map((item, i) =>
                i === index ? { ...item, selectedImageType: newType } : item
            )
        );
    };
    return (
        <>
            <ManagerContainer>
                {sections.map((item, index) => (
                    <SectionFormatter
                        index={index}
                        scriptValue={item.script}
                        onScriptChange={(text) => handleScriptChange(index, text)}
                        imageType={item.selectedImageType}
                        onImageTypeChange={(e) => handleImageTypeChange(index, e.target.value)}
                        realImageUrl={item.realImage}
                        aiImageUrl={item.aiImage}
                    />
                ))}
            </ManagerContainer>

            <SubmitButtonWrapper>
                <SubmitButton onClick={() => {
                    const result = sections.map(({ script, selectedImageType }) => ({
                        script,
                        imageType: selectedImageType === 'real' ? 0 : 1
                    }));
                    onSubmit(result);
                }}>
                    영상 생성
                </SubmitButton>
            </SubmitButtonWrapper>
        </>
    );
}

export default VideoSourceManager;